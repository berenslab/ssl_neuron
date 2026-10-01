# PLAN: getting GraphDINO / GICLMorph to beat the depth baseline

Status as of 2026-10-01. Written down so the reasoning survives the next
cluster session. Everything under "Diagnosis" is a hypothesis unless it says
"measured".

## 1. Where we are

The first serious GraphDINO run (`config.json`: 300 epochs, 2952 train /
369 val / ~370 test cells, `num_classes` 1000, batch 16, ~1.3 it/s, ~13 h) is
**not working**, although its loss looks fine.

Measured at a mid-run checkpoint (`07`, val queries, 5-NN balanced accuracy):

| | value |
|---|---|
| depth-profile baseline (no learning) | 0.814 (stratified CI to be re-read from `eval_results_val.csv`) |
| GraphDINO, paired difference to baseline | **-0.324**, 95% CI [-0.378, -0.266] |

- Per-class recall: GraphDINO is below the baseline on almost every class,
  worst on types separated by fine morphology (small-RF and transient types,
  `ON vertical OS small RF`, `EW5to`, `OFF sustained EW3o`). It matches or beats
  the baseline only on a few (`ON alpha`, `F-mini-OFF`, `OFF vertical OS`).
- t-SNE of the embedding is organised almost entirely by mean node z. Classes
  separate only where they differ in depth; types inside one depth band overlap.

Training diagnostics (`metrics.csv` / console), measured:

| column | value | meaning |
|---|---|---|
| `loss` | 6.76 -> 0.27 | looks like learning, is not (see below) |
| `H_t` (teacher entropy) | 5.5 -> 0.02 (ln K = 6.91) | teacher is one-hot for every cell |
| `H_marg` (entropy of batch-mean teacher) | 6.1 -> **~1.5, flat from epoch 10** | only ~4-5 of 1000 prototypes in use (`exp(1.5) ~ 4.5`) |
| `emb_eff_rank`, `emb_mean_cos` | ~5.5, ~0.22 | embedding spans ~5 directions |
| `\|g\|` | ~10 against `grad_clip` 3 | clipping active on every step |

Conclusion: **partial collapse onto ~5 coarse depth bands.** Not the useless
uniform collapse (cos 0.22, rank 5.5), but far too coarse for ~35 scored types.

## 2. What the data tells us about the benchmark

- 3691 cells, 2915 labeled, 39 types, 35 scored (`>= 5` labeled train cells).
  Rare types have 0-3 val queries each.
- Labels are noisy (~10% wrong even for `both_strong`), and some types have no
  label yet. Practical ceiling for any model is ~0.9 balanced accuracy.
- Depth + field size alone already reach ~0.81 (val) / ~0.86 (leave-one-out).
  Retinal ganglion cell types are largely defined by stratification depth. So
  the bar is "beat the depth baseline", not "beat chance" (0.029), and the
  headroom is only ~0.05-0.10. Differences between models must be judged on
  **paired** differences, not on two overlapping intervals.

## 3. Diagnosis (hypotheses, not yet tested)

1. **Unbounded logits.** The DINO projector output is unbounded, so the student
   can make the teacher one-hot just by growing logit scale. Temperatures and
   centering then barely matter. Standard DINO bounds this with an
   L2-normalised bottleneck and a weight-normalised last layer.
2. **Weak centering.** The center is estimated from only 2 x 16 teacher
   outputs per step (`center_avg` 0.9).
3. **`student_temp` is 0.9**, hard-coded in `graphdino.py` (usual DINO value:
   ~0.1), and not exposed in the config.
4. **Depth shortcut.** Both views share the same depth frame (only 0.4 z-jitter),
   so "same z" solves the task and nothing rewards separating cells inside a
   depth band.

## 4. Plan

Stop the current 300-epoch run (it will not recover; `H_marg` has been flat
since epoch 10). Then run **short experiments of ~25 epochs (~1 h each)**.

Judge each experiment on, in this order:
1. `H_marg` stays above ~4 (and `H_t` neither ~0 nor ~ln K);
2. `07` with `GICLMORPH_EVAL_VIEWS=1` (a few minutes, embeddings are cached):
   the paired difference to the depth baseline in `eval_paired.csv`;
3. only then the leave-one-out and per-class numbers.

Experiments, in priority order:

| # | change | cost |
|---|---|---|
| 1 | bounded logits: DINO-style head (L2-normalised bottleneck + weight-normalised last layer) as a config option, current head stays the default | model change, small |
| 2 | expose `student_temp` in `config['model']`; try 0.1, with `teacher_temp` 0.04-0.1 | trivial |
| 3 | larger batch (32 / 64 if the GPU memory allows; 7.3 GB at 16 today) for better centering | config only |
| 4 | weaken the depth shortcut: more z-jitter or a per-view random depth shift | conflicts with `00_dataset_spec.md` (no z-translation); last resort |
| 5 | `grad_clip` 10-15 so clipping does not dominate | config only |
| later | Sinkhorn-Knopp centering with fewer prototypes (64-128); `num_classes` 256 | only if 1-3 are not enough |

Also worth a look if training is slow: 1.3 it/s is probably CPU-bound by the
augmentation (`subsample_graph`) in 4 workers. Check `nvidia-smi`; if GPU
utilisation is well under ~80%, raise `data.num_workers`.

GICLMorph (`06`) is **not started** and costs about twice as much per step.
Do not start it until a GraphDINO configuration beats the baseline; then run it
with the same trainer/optimizer/data blocks, so the comparison is like for like
(`cmid_weight: 0` is the control).

## 5. Evaluation protocol

- Splits (from `01`): train 80% / val 10% / **test 10%, frozen** (`test_ids.npy`
  is reused, never redrawn; delete it by hand to start over and retrain
  everything). All stratified by celltype, unlabeled cells as one stratum.
- Tune on **val**. Test is only loaded and scored with `GICLMORPH_EVAL_TEST=1`
  and only for the final GraphDINO vs GICLMorph comparison. No checkpoint is
  picked by val loss; use the final one.
- Headline numbers: 5-NN balanced accuracy on `both_strong` queries and on all
  queries, leave-one-out over train+val (lower variance while tuning), and the
  **paired difference** between runs with a stratified-bootstrap interval. A
  difference whose interval contains 0 is not a result.
- Report rare classes (< ~20 cells) per class, not inside the average.

## 6. How to read the training log

`Epoch N | it | lr | loss | KL | H_t/lnK H_s H_marg | |g| | val | rank cos | it/s | ETA | GPU`

- `H_t` ~ ln K: uniform collapse. `H_t` ~ 0 with `H_marg` ~ 0: one-prototype
  collapse. `H_t` ~ 0 with `H_marg` ~ 1-2: **the failure seen here.** Healthy:
  `H_t` well below ln K and `H_marg` close to ln K.
- The DINO loss is a cross-entropy bounded below by `H_t`; it can fall while
  nothing useful is learned. Read `KL = loss - H_t` and `H_marg` instead.
- `rank` / `cos` (every `eval_every` epochs, on val): `cos` -> 1 or `rank` -> 1
  means the embedding collapses to a point or a line.
- `metrics.csv`, `run_info.json`, `config.json` and `last.pt` (resume with
  `trainer.resume: true`) are in the checkpoint directory.

## 7. Open questions

- Which checkpoint was the `-0.324` measured on? (If it was early, the picture
  may shift a little, but `H_marg` flat since epoch 10 argues against it.)
- Is a coarse depth-only embedding acceptable as a *baseline-level* result, or
  must the learned model add information beyond depth to be worth publishing?
  This decides whether experiment 4 (touching the depth augmentation) is allowed.
- Types without labels (e.g. `DAC`, `ON SAC`, `m8X`): can a held-out test set
  say anything about them? Probably only qualitatively (do they cluster?).
