# GraphDINO on eyewire2 RGCs — dataset & augmentation spec

Status: agreed 2026-09-22 and implemented in this folder (§9). Everything
marked **Assumed** is a recommendation adopted without explicit sign-off —
cheap to revisit, but write down the reason if you change it.

This document explains why the eyewire2 retina pipeline (`01`–`04` in this
folder) cannot use GraphDINO's stock configuration, and fixes the coordinate
frame and augmentation policy that the rest of the pipeline implements.

---

## 1. Why this dataset is not the Allen dataset

GraphDINO's defaults were tuned for cortical neurons from the Allen Brain
Atlas: full 3D morphologies whose informative axis is distance from the pia,
reconstructed one cell at a time. Retinal ganglion cells from eyewire2 differ
in four ways that each invalidate part of the stock recipe.

### 1.1 Only the dendrites are trustworthy

Axons (SWC type 2) in these skeletons are incomplete and unreliable: some cells
have one, most don't, and where one exists its extent reflects how far the
reconstruction got, not the cell's biology. Only the soma (type 1) and
dendrites (type 3) carry signal.

#### Consequences

- Axon nodes are dropped in preprocessing, before anything else sees them. No
  feature and no augmentation may reintroduce them.
- Axon removal can *disconnect* the graph — not only when the axon is a clean
  leaf subtree hanging off the soma (harmless), but whenever a few nodes along
  a dendritic path are mislabeled as axon (plausible, given that the labels are
  unreliable). The connectivity check and repair therefore run **after** axon
  removal, not only before it.
- Any downstream claim about "arbor size" means *dendritic* arbor size.

### 1.2 Depth carries most of the celltype signal

What separates RGC types is largely *where in the IPL they stratify* — i.e. the
1-D profile of dendritic length binned over depth. The z coordinate of these
skeletons is already projected into a normalized depth space shared across
neurons (pywarper's conformal flattening, after Sümbül et al. 2014), so z is
directly comparable between cells and is the single most informative axis.

**Consequences** — the z axis is close to sacred:

| Operation on z | Verdict | Why |
| --- | --- | --- |
| Translation | **Never** | Shifts the cell to a different stratification depth — i.e. to a different celltype. |
| Flip / mirror | **Never** | Swaps ON and OFF. That is precisely the distinction we want the embedding to make. |
| Scaling | **Never** | Changes how many strata the arbor spans. |
| Rotation out of plane | **Never** | Mixes depth into xy; only valid if the warp were an isometry, which it is not. |
| Soma-centering | **Never** | Would re-reference each cell to its own soma and destroy the shared frame (see §6.2). |
| Small jitter | **Yes, bounded** | Must stay at or below the cross-cell alignment error of the warp (§4). |

### 1.3 Rotation invariance in xy is imposed deliberately

The retina is *not* rotation invariant — dendritic asymmetry has a real
preferred direction for some types. We impose xy rotation invariance anyway,
accepting that orientation subtypes get collapsed, because orientation is easy
to recover post hoc from a per-cell asymmetry vector (§8). The benefit is a far
better-conditioned learning problem on a small dataset.

### 1.4 xy shape matters; xy position does not

Dendritic field diameter and branch density in the xy-plane are real,
discriminative features (alpha subtypes differ substantially in field size).
Absolute xy position is meaningless for typing — it only matters for mosaic
analysis, which happens in a later, separate stage.

#### Consequences

- xy translation is a valid augmentation; xy scaling essentially is not
  (§4 allows ±5% as noise absorption only).
- The xy rotation must be exactly norm-preserving. The stock implementation is
  not, which silently destroys this signal — see §6.1.
- Absolute soma position is saved to a sidecar file during preprocessing so the
  later mosaic stage doesn't require re-running the pipeline. **Assumed.**

---

## 2. Coordinate frame contract

Everything downstream of `01_preprocess_data.py` assumes:

| Axis | Origin | Frame | Units |
| --- | --- | --- | --- |
| x, y | soma at (0, 0) | per-cell, soma-centered | µm |
| z | **untouched** | shared warped IPL depth frame | see open item §7.1 |

Plus the layout `GraphDataset` already requires: soma is node 0, the graph is a
single connected component, axons are removed, `features.npy` is N×3 (xyz).

---

## 3. Dataset scope and QC

- **Training set**: all QC-passing RGC skeletons, labeled or not. DINO needs
  volume and does not use labels; the labeled cells serve as the evaluation
  subset. (Previously the pipeline filtered to the three alpha types.)
  Concretely, selection is `cellclass_final == 'RGC'`, further masked by
  `valid_cellclass_final` and `valid_status` where those columns exist — the
  same consensus-label columns `eyewire2-datajoint`'s
  `prepare_manuscript_df.py` produces and `plot_RGCs.ipynb` selects on.
- **QC rule**: drop skeletons below 1000 nodes. **Assumed.** In the earlier
  113-cell snapshot the 10th percentile of xy extent was 28 µm, which is a
  broken or partial reconstruction, not a small cell. Since arbor size is now a
  load-bearing feature, a truncated arbor is worse than no cell at all. On the
  current 184-cell snapshot this threshold drops nothing (smallest skeleton:
  1587 nodes), so it costs only the cells it is meant to catch.
- **Volume-boundary truncation**: cells whose dendrites leave the imaged volume
  should be excluded rather than kept as small cells — see §7.2.

---

## 4. Augmentation policy

Two independently augmented views per cell, as in stock GraphDINO. What
changes is which transforms are allowed and with what magnitude.

### Graph-topology augmentations (unchanged in kind, retuned)

| Augmentation | Setting | Rationale |
| --- | --- | --- |
| Random branch deletion | `n_drop_branch: 5` (was 10) | Keeps views genuinely different without routinely amputating enough arbor to distort field size or the depth profile. |
| Subsample to fixed N | `n_nodes: 512` (was 200) | The depth profile the model sees *is* the z-histogram of these N nodes. At 200 nodes a 900 µm alpha arbor gets one node per ~20 µm; 512 gives ~2.5× the profile resolution at ~6.5× the attention cost (O(N²)). |
| Load-time cache subsample | `cache_nodes: 2048` (was hardcoded 1000) | Must stay comfortably above `n_nodes` or per-view subsampling has nothing left to vary. |

### Position augmentations

| Augmentation | Setting | Rationale |
| --- | --- | --- |
| Rotation about z | uniform angle in [0, 2π), proper SO(2) | §1.3. Must be a true rotation — see §6.1. |
| Mirror in xy | random x-flip, p = 0.5 | Dendritic arbors have no meaningful xy chirality; with the rotation this gives full O(2) invariance and doubles view diversity for free. |
| Isotropic xy scaling | ±5% | Absorbs reconstruction and warping scale noise while leaving the large between-type size differences intact. Applies to x and y only; z is never scaled. |
| Jitter | σ = (1.0, 1.0, 0.4) | xy at reconstruction-noise scale; z at ~1% of IPL thickness, blurring sub-micron depth without smearing across strata. |
| Translation | σ = (10.0, 10.0, **0.0**) | Forces the model off absolute xy position. The z component must be exactly zero (§1.2). |

Order of operations: rotate → mirror → scale → jitter → translate. Scaling
precedes jitter so that jitter magnitude stays in absolute units.

### Explicitly rejected

- **z-flip, z-translation, z-scaling** — §1.2.
- **Anisotropic xy scaling or shear** — distorts field shape, which is signal.
- **Random xy wedge/sector removal** — mimics incomplete reconstruction, but
  corrupts field size and symmetry, which are the features we are protecting.
- **Dropping a contiguous z-slab** — deletes exactly the depth profile.
- **Arc-length resampling of nodes in preprocessing** — considered and
  rejected; skeliner's native node spacing is kept. See the caveat in §7.3.

---

## 5. Features

`features.npy` stays N×3 (xyz only), `feat_dim: 3`. **Assumed.**

Radius is the tempting addition — alpha cells genuinely have thick primary
dendrites — but EM radius is confounded by segmentation and skeletonization
quality, so it risks encoding reconstruction batch rather than biology. It is
kept in the SWC parse and available for a v2 ablation. Soma size is likewise
left out; the soma stays a single node at the origin and its size is better
used post hoc.

---

## 6. Upstream issues this required fixing

Three defects in the stock code are actively harmful for this dataset. All
three are silent — nothing crashes, the model just learns the wrong invariances.

### 6.1 `rotation_axis: "z"` is not a rotation

`ssl_neuron/utils.py:rotate_graph` draws a random *3D* rotation matrix and then
zeroes the z row and column. z is correctly preserved, but the remaining 2×2 xy
block of a random 3D rotation is **not orthogonal**. Measured over 20 000 draws:

- median xy *area* factor 0.50 (5th–95th percentile: 0.05–0.95),
- 50% of draws squash the arbor by more than 2× along some xy direction.

So the stock "rotation" randomly rescales and shears the arbor — precisely the
xy size and density signal §1.4 says we must protect. Replaced with a proper
SO(2) rotation by a uniformly drawn angle.

### 6.2 Preprocessing soma-centered all three axes

`features[:, :3] -= features[soma_id, :3]` subtracted soma z along with soma
xy, re-referencing every cell to its own soma and discarding the shared depth
frame that §1.2 depends on. Now xy-only.

**This invalidates every previously preprocessed skeleton.** The existing
`data/skeletons/*` were written with soma-centered z; `01` must be re-run on the
cluster before training.

### 6.3 Jitter and translation were isotropic

`jitter_var: 1` and `translate_var: 10` applied the same magnitude to z. A
10 µm z-shift on a ~35 µm-thick IPL moves a cell across strata. Both are now
per-axis vectors.

---

## 7. Open items

### 7.1 What unit is z in?

The warped z is either µm or percent-IPL-depth. On the current snapshot the
per-cell z extent has median 30.4 and the pooled node-z distribution spans
about 34 units (1st–99th percentile), which reads as an IPL thickness in µm —
but percent-IPL cannot be ruled out from soma-centered data alone, since every
cell's depth is currently referenced to its own soma.

This sets the scale of the jitter's z component: σ = 0.4 is right if z is µm,
and should become ≈ 1.0 if z is percent-IPL. `02_visualize_data.py` prints the
across-cell z distribution and the pooled depth histogram; once `01` has been
re-run with xy-only centering, the IPL band structure should be visible across
cells and settles this. Do it before the first real training run.

### 7.2 Is there a truncation flag? — partly answered

There is no explicit truncation flag in `df_all_neurons`. What it does carry
(from `prepare_manuscript_df.py`) is `hull_points`, `hull_diameter`,
`hull_perimeter` and `height`, which are the natural handle: a cut arbor shows
up as a hull whose boundary runs along the volume border. Wiring that check in
is still to do; until then, truncated cells are only caught by the node-count
floor, which will miss the ones that are large but clipped.

### 7.3 Is skeliner's node spacing uniform? — **resolved: resampling not needed**

Spacing is *not* uniform: over 20 cells, edge length has median 0.355 µm,
IQR 0.29–0.56 and a coefficient of variation of 0.77.

That spread turns out not to matter, because it does not vary with depth.
Comparing each cell's node-weighted depth profile against its length-weighted
one across all 184 cells of the current snapshot:

- correlation: median 0.9994, worst cell 0.9935,
- L1 distance: median 0.039, p90 0.073, worst 0.124 (on profiles summing to 1),
- edges longer than 5× the median: 0.09% of edges, worst cell 0.9%.

So the node histogram the model sees is a faithful stand-in for dendritic
length density, and keeping skeliner's native spacing (§4) is safe. Re-check
this after re-preprocessing — the measurement above is from skeletons written
by the old, soma-z-centered pipeline, which does not affect spacing but does
mean the cell set may shift.

### 7.4 How accurate is the warp across cells?

The z-jitter magnitude should sit at roughly the cross-cell alignment error of
pywarper's flattening. σ_z = 0.4 assumes that error is at or below ~0.5 µm. If
it is closer to 1–2 µm, raise the jitter to match, or the model will latch onto
depth differences finer than the data supports.

### 7.5 Training length

`max_iter: 5000` was set for a ~100-cell run. With the full RGC set this is
only a few hundred epochs; revisit once the final cell count is known.

---

## 8. Evaluation plan

Primary: k-NN classification accuracy on `celltype_final` over the labeled
subset, using embeddings from the frozen student. **Assumed** as the headline
number over a linear probe, because it makes no assumption about the embedding
being linearly separable.

Sanity check: agreement between embedding clusters and the classical z-density
profiles. If the embedding is doing its job, cells that cluster together should
have similar stratification profiles — and if they don't, that is more
interesting than the accuracy number.

Post-hoc stages, deliberately outside the model:

- **Orientation subtypes** (§1.3): fit a per-cell arbor asymmetry vector, then
  sub-cluster within each embedding cluster by its direction.
- **Mosaics** (§1.4): uses the absolute soma positions saved during
  preprocessing.

---

## 9. Where this lives

| File | Role |
| --- | --- |
| `augment.py` | The position augmentations of §4, plain numpy so they can be inspected without the `torch` extra. Refuses a z-translation outright. |
| `preprocessing.py` | SWC parsing, axon removal, sparse connectivity repair, QC, xy-only centering. The pure functions behind `01`. |
| `dataset.py` | `RetinaGraphDataset`, a `GraphDataset` subclass that swaps in those augmentations, plus a matching `build_dataloader`. |
| `config.json` | The §4 parameter values, under `data.augment`. |
| `01_preprocess_data.py` | Driver: cell selection, the preprocessing loop with a QC breakdown, `cell_meta.csv`, train/val split. |
| `02_visualize_data.py` | Sanity checks, and the measurements that settle §7.1 and §7.3. |
| `03_train_graphdino.py` | Training, now using `RetinaGraphDataset`. |

Changes outside this folder, all backward compatible (the Allen config still
runs unchanged):

- `ssl_neuron/datasets.py`: `jitter_var`, `rotation_axis` and `translate_var`
  became optional; the load-time node cap became the `cache_nodes` config key
  instead of a hardcoded 1000; `build_dataloader` takes a `dataset_cls`.

The stock `rotate_graph`, `jitter_node_pos` and `translate_soma_pos` in
`ssl_neuron/utils.py` are untouched and still used by the Allen pipeline. They
are simply not used here — see §6.
