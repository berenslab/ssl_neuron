# GICLMorph on eyewire2 RGCs — method & adaptation spec

Status: written 2026-09-23, smoke-tested on CPU on the local 184-cell alpha
snapshot (§6) but **not yet trained on the cluster**. Everything marked
**Assumed** is a choice made without sign-off — either because the paper leaves
it open or because the paper's choice is wrong for this dataset. Cheap to
revisit; write down the reason if you change it.

The paper: Hao, Li, Wu, Sheng & Qu (2026), *GICLMorph: Self-supervised 3D
neuronal morphology representation via graph-image contrastive learning*,
Expert Systems With Applications 320, 132059. No code was released (the paper
says data "on request"; a search found no repository), so everything here is a
reimplementation from the text.

This document assumes `00_dataset_spec.md` (the GraphDINO adaptation). Its
coordinate frame, QC and augmentation policy apply unchanged to the graph
branch here, and are not repeated.

---

## 1. What GICLMorph is, and why it lives in this repo

GICLMorph = **GraphDINO + an image branch**.

| Paper component | Paper setting | Here |
| --- | --- | --- |
| Graph encoder | Graph transformer with AC-attention, 7 blocks, 8 heads, dim 32 | `graphdino.GraphTransformer`, same config — it *is* GraphDINO's encoder |
| IMID loss (§3.3, eqs. 5–8) | Cross-view self-distillation, EMA teacher (m = 0.999), centered target, τ_t < τ_s | `graphdino.GraphDINO`, unchanged |
| Image encoder (eq. 9) | ResNet-50 (2048-d) + projection head g_δ | `giclmorph.ImageEncoder` |
| CMID loss (§3.4, eqs. 10–13) | Symmetric InfoNCE between z_i (graph, view 1) and h_i (image), cosine similarity; negatives are the other cells in *both* modalities | `giclmorph.cmid_loss` |
| Total loss (eq. 14) | L = L_IMID + γ·L_CMID, γ = 0.4 best | `giclmorph.cmid_weight = 0.4` |
| 2D views (§3.2) | PCA-guided canonical projections, K = 6 | `projection.py`, see §2 |
| Optimizer | Adam, weight decay 0.01, lr 1e-4, linear warm-up then ×0.5 every 20k steps, 1e5 steps, batch 16 | `Trainer`'s schedule already is exactly this (decay every `max_iter // 5`); see §4 |
| Graph size | Subsampled to 1000 nodes by removing non-branching nodes | 512 nodes, GraphDINO eyewire2 setting (`00_dataset_spec.md` §4) — **Assumed** |

Because the graph branch is literally GraphDINO, setting `cmid_weight = 0`
recovers GraphDINO **exactly**, with the same data, augmentations and budget.
That is the paper's "Intra-only" ablation, and it is the control every
GICLMorph number here must be compared against. It is also why this is not a
separate repo: a separate one would have had to duplicate the graph
transformer, the graph augmentations and the whole eyewire2 preprocessing.

---

## 2. The PCA-guided projection cannot be used as-is on RGCs

The paper's recipe (§3.2): resample the skeleton at fixed geodesic spacing,
center on the centroid, 3D PCA, rotate so that PC1 is the image's vertical
axis, then for view k rotate about PC1 by θ_k = 2π(k−1)/K and project onto the
plane (PC1, rotated PC2). Every neuron is normalized to [−1, 1].

That recipe is built for cortical cells, whose PC1 is the apical/pia axis — the
biologically meaningful axis *is* the axis of maximum variance. On RGCs it
breaks four ways:

1. **PC1 and PC2 both lie in the xy-plane.** A dendritic arbor is ~300 µm wide
   and ~35 µm deep, so depth is PC3. The paper's views are then en face (θ = 0,
   180°) or obliques dominated by PC2 (θ = 60°, 120°, …). With K = 6 there is no
   θ = 90° view at all, so **depth — the primary celltype signal — never
   appears in any image.** `05_render_projections.py`'s last plot shows this on
   real cells.
2. **Centroid centering discards absolute depth.** Same defect as
   `00_dataset_spec.md` §6.2: it re-references each cell to itself.
3. **Per-cell [−1, 1] scaling discards dendritic field size**, which separates
   alpha from midget-like types (`00_dataset_spec.md` §1.4).
4. **Eigenvector signs are arbitrary.** Along a depth axis a sign flip swaps ON
   and OFF. The paper does not say how it breaks the tie.

Translated to RGCs, the paper's own idea — *rotate about the biological axis,
project onto planes containing it* — asks for depth as that axis, not PC1. So
`projection.retina_views` (**Assumed**, the default `mode: "retina"`):

| | Paper (`mode: "paper"`) | Retina (`mode: "retina"`) |
| --- | --- | --- |
| Biological axis (image vertical) | PC1 of 3D PCA | **z (depth)**, fixed |
| Vertical scale | per cell, [−1, 1] | **global** `z_range` (µm), never centered |
| Reference direction for θ = 0 | PC2 | e1 = principal axis of the **xy-only** PCA |
| View k | plane (PC1, cosθ PC2 + sinθ PC3) | side view: horizontal = e1 rotated in-plane by θ_k |
| Horizontal scale | per cell | **global** `xy_half_extent` (µm), soma at center |
| Extra views | — | one en-face view (e1, e2), same global scale (`en_face: true`) — **Assumed** |
| Sign convention | not specified | each axis flipped so the projection's third moment is positive |

Rotating only about z never mixes depth into xy, so the images obey the same
invariance contract as the graph augmentations. The side views carry
stratification (rows = IPL depth, identical across cells), the en-face view
carries field shape and size. The paper's projection stays available as
`mode: "paper"` — for an ablation, and for any non-retina dataset.

Two consequences worth knowing:

- **The in-plane reference e1 is arbitrary for round arbors** — the paper's
  "PCA failure case", anisotropy AI < 0.1 (its Table 2). `05` prints the
  fraction (2.2% on the local alpha snapshot). For those cells the side views
  are still consistent in depth, which is what matters most; only *which*
  side you are looking from is random.
- `xy_half_extent` and `z_range` default to `null`, which makes `05` derive them
  from the data once (99th-percentile arbor radius; 0.1/99.9 depth percentiles
  padded by 5%) and store them in `data/projections_meta.json`. They are dataset-level
  constants, never per-cell.

### Rendering

The paper says "grayscale images", 224 × 224, and nothing else. Here
(**Assumed**): the resampled points (0.5 µm spacing) are binned into pixels, so
a pixel's value is the cable length through it, log-scaled against a *global*
`max_count` so brightness is comparable across cells. Stored as uint8 in
`skeletons/<cell_id>/projections.npy` (V × 224 × 224, V = 7), ~350 KB per
cell. Grayscale is replicated to three channels for the ResNet.

---

## 3. What the paper leaves open, and what was chosen

| Question | Choice | Why |
| --- | --- | --- |
| Which graph vector enters CMID? | z = g_θ(y), the DINO projector output (`cmid_graph_input: "proj"`) | Fig. 1 and eq. 10 use the same z^{t1} for both losses. `"cls"` (a separate head from the CLS embedding into a shared 128-d space, as in CrossPoint) is the obvious alternative — **Assumed**, first ablation if CMID fights IMID. |
| Which graph view enters CMID? | View 1 only | Eq. 10 uses z^{t1}; view 2 only sees IMID. |
| Which image per step? | One of the V views, uniformly at random | The paper renders K views per cell but does not say how they are fed. **Assumed.** |
| CMID temperature τ | 0.1 | Not given; CrossPoint's value. **Assumed.** |
| "Adam, weight decay 0.01" | AdamW, 0.01 | Coupled L2 of 0.01 under Adam is unusually strong for a transformer; decoupled decay is the likely intent. `optimizer.name: "adam"` gives the literal reading. **Assumed.** |
| ResNet init | Random (`image_pretrained: false`) | Not stated. ImageNet weights need a download on the cluster; worth trying. |
| Image-branch augmentation | None | None described; the choice among fixed views is the only variation. |
| Graph node features | xyz only | The paper uses x, y, z, radius, type. Radius is confounded by EM segmentation quality and type is constant after axon removal (`00_dataset_spec.md` §5). |
| Graph augmentations | The retina policy of `00_dataset_spec.md` §4 | The paper also uses full 3D rotations and axial reflections, which would destroy the depth frame. |

---

## 4. Budget

The paper trains for 10⁵ iterations at batch 16. That is the default in
`config_giclmorph.json`, versus 5000 in the GraphDINO `config.json` (whose
`00_dataset_spec.md` §7.5 already flags it as too short). **Compare like with
like**: train the GraphDINO control with `06` and `cmid_weight: 0` rather than
with `03`, or give `03` the same `max_iter`. Otherwise a GICLMorph win may just
be 20× more training.

A step costs roughly GraphDINO's plus a ResNet-50 forward/backward at 224 px,
batch 16. The ResNet only sees one image per cell per step.

---

## 5. Evaluation (`07_evaluate_embeddings.py`)

Same protocol for every run (GraphDINO from `03`, GICLMorph from `06`, extra
runs via `GICLMORPH_EVAL_RUNS`), on the frozen student's 32-d CLS embedding,
with no augmentation (averaged over `n_eval_views` random subsamples to
`n_nodes`):

- **k-NN balanced accuracy** (cosine, k = 5) — headline number, per
  `00_dataset_spec.md` §8. Database = labeled cells in `train_ids`, queries =
  labeled cells in `val_ids`, which SSL training never sees.
- **Linear-probe balanced accuracy** (the paper's "SSL-linear probing").
- The paper's clustering metrics, with celltype as the cluster assignment:
  **silhouette** and **Davies–Bouldin**, plus k-means ARI.
- **E-Top10**, the paper's collapse diagnostic (Table 8).
- A **depth-profile baseline** — node-depth histogram + field size, no learning
  — scored with the same k-NN. On RGCs this is the floor to beat.

The paper's Intra-CD / Inter-CD are not reported: they are raw distances, so
they depend on the embedding's scale and are not comparable across models.

---

## 6. Smoke test

Done on CPU (Windows) on 2026-09-23, on the 184 local alpha-type skeletons (3
classes; the old soma-z-centered snapshot). A pipeline check, not a result.

- `cmid_loss` matches a literal loop over eqs. 10–13 to 1e-6.
- With `cmid_weight: 0`, `GICLMorph` returns *bit-identical* loss to a
  `GraphDINO` with the same weights. Gradients reach the student and the image
  branch; the teacher gets none.
- `05` renders 184 cells × 7 views in ~25 s; <1% of cable is clipped for all
  but 2 cells.
- `06` (`n_nodes` 128, ResNet-18, 20 iterations) and `07` run end to end.
  Twenty iterations say nothing about whether training works; the loss trend
  and every accuracy number have to come from the cluster.
- **This local set cannot rank models.** The depth-profile baseline already
  scores 0.96 balanced 5-NN accuracy on it, because ON, OFF-transient and
  OFF-sustained alpha are separated by depth alone. The full ~28-type set is
  the benchmark.

On Windows the scripts hang in `GraphDataset.__init__`: upstream's
`multiprocessing.Manager()` fails to start under `spawn`. The smoke test
patched it out from a driver script; nothing in the repo was changed for it,
because on the Linux cluster (`fork`) and in Jupyter this does not arise.

---

## 7. Open items

### 7.1 Does the image branch add anything a depth histogram doesn't?

On RGCs the side views mostly encode the stratification profile, which the
graph already sees through node z. The paper's argument (GNNs miss global
geometry) is weaker when the global signal is a 1-D profile along an axis every
node carries directly. The comparison that settles it: GraphDINO (γ = 0) vs
GICLMorph (γ = 0.4) vs the depth-profile baseline, same budget, in `07`.

### 7.2 γ and K were tuned on cortex

γ = 0.4 and K = 6 are the paper's optima on ACT/BIL/BBP. Sweep γ ∈ {0.1, 0.4,
0.9} before believing any single run.

### 7.3 Is the en-face view needed?

`en_face: false` gives the paper's K side views only. Field size is already in
the side views' horizontal extent, so the en-face view may be redundant.

### 7.4 Inherits every open item of `00_dataset_spec.md` §7

In particular §7.1 (is z µm or %IPL — it sets how `z_range` reads) and the
requirement to re-run `01` on the cluster: the local skeletons still have
soma-centered z, which `05` warns about.

### 7.5 Memory and step time are unmeasured

Nothing has run on a GPU yet. ResNet-50 at 224 px, batch 16, on top of
GraphDINO at 512 nodes should fit comfortably on a 24 GB card, but that is an
estimate. Log `torch.cuda.max_memory_allocated()` and the time per iteration
on the first cluster run: 10⁵ iterations may take much longer than expected.

### 7.6 The evaluation split is small and not MorphoGNN's — partly addressed

`07` queries the labeled cells in `val_ids`. `01` now stratifies that 10% by
celltype (`00_dataset_spec.md` §3), so every type gets its share of queries.
That is 291 labeled val cells on the full set, 1–24 per scored type. Before,
a plain random draw could leave a small type with none. Balanced accuracy is
still noisy for the small types: 2–3 queries each. The split is also still
not the stratified 70/15/15 of `MorphoGNN/eyewire2`, so the numbers are not
directly comparable with the supervised MorphoGNN test accuracy. Options: a
larger `VAL_FRACTION` in `01`, or cross-validated k-NN over all labeled cells
in `07` (the embedding never saw labels, so this does not leak).

### 7.7 No checkpoint selection

`07` evaluates the last checkpoint of each run. There is no early stopping,
although the paper mentions it. With `save_ckpt_every` checkpoints on disk, a
k-NN curve over checkpoints is cheap to add to `07` if the last one turns out
worse than an earlier one.

### 7.8 `04_visualize_results.py` only loads GraphDINO checkpoints

A GICLMorph `state_dict` nests the graph model under `dino.` and adds
`image_encoder.`. `04` builds a plain `GraphDINO`, so it cannot load one. Use
`07` for GICLMorph runs, or load the checkpoint with
`giclmorph.create_model(config)`; `model.student_encoder` then works as in `04`.

---

## 8. Where this lives

| File | Role |
| --- | --- |
| `ssl_neuron/projection.py` | Resampling, PCA frames, `paper_views`, `retina_views`, rendering. numpy only. |
| `ssl_neuron/giclmorph.py` | `ImageEncoder`, `cmid_loss`, `GICLMorph` (wraps a `GraphDINO`), `create_model`. |
| `ssl_neuron/datasets.py` | `GraphImageDataset`: `GraphDataset` + one random projection per item. |
| `ssl_neuron/train.py` | `GICLTrainer`: same schedule, logs IMID/CMID, writes `history.csv`. |
| `eyewire2/dataset.py` | `RetinaGraphImageDataset`, `build_giclmorph_dataloader`. |
| `eyewire2/config_giclmorph.json` | Everything above, in one place. `model`/`data` match `config.json`. |
| `eyewire2/05_render_projections.py` | Renders `projections.npy` per cell; frame constants; paper-vs-retina plot. torch-free. |
| `eyewire2/06_train_giclmorph.py` | Training; copies its config into the checkpoint dir. |
| `eyewire2/07_evaluate_embeddings.py` | k-NN / linear probe / clustering for GraphDINO and GICLMorph runs, plus the baseline. |

Changes outside this, all backward compatible (GraphDINO runs unchanged):

- `graphdino.GraphDINO.forward` takes `return_student=False`; when true it also
  returns the student's embeddings and projections.
- `train.Trainer` reads optional `optimizer.name` (`adam`/`adamw`, default
  `adam`) and `optimizer.weight_decay` (default 0), and always saves the final
  epoch's checkpoint, not only multiples of `save_ckpt_every`.

```bash
uv sync --extra torch
python ssl_neuron/eyewire2/01_preprocess_data.py      # if not done yet
python ssl_neuron/eyewire2/05_render_projections.py   # torch-free
python ssl_neuron/eyewire2/06_train_giclmorph.py      # GPU
python ssl_neuron/eyewire2/07_evaluate_embeddings.py
```

`05`–`07` take `GICLMORPH_CONFIG`, `GICLMORPH_DATA` and (06) `GICLMORPH_CKPTS`
environment variables, so a sweep over γ is a loop over config files and
checkpoint directories without editing anything.
