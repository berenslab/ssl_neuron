# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SSL-Neuron / GraphDINO: self-supervised graph representation learning for neuronal morphologies (skeleton graphs), from Weis et al. 2023 ("Self-Supervised Graph Representation Learning for Neuronal Morphologies", TMLR). GraphDINO is a graph-transformer trained with a DINO-style student/teacher self-distillation objective on augmented views of the same neuron skeleton.

The `ssl_neuron/eyewire2/` subfolder adapts the original Allen Brain Atlas (ABA) pipeline to eyewire2 retina skeletons produced by the sibling `skeliner` repo. **`ssl_neuron/eyewire2/00_dataset_spec.md` is the authority for that adaptation** — what makes retinal ganglion cells different from ABA cortical cells, which augmentations are therefore legal, and which stock behaviors had to be replaced. Read it before changing anything in that folder; several of the defaults there look arbitrary but are load-bearing.

The repo also carries **GICLMorph** (Hao et al. 2026, reimplemented from the paper — no official code exists): GraphDINO plus a ResNet branch on PCA-guided 2D projections, aligned by a cross-modal InfoNCE loss. It wraps an unchanged `GraphDINO`, so `giclmorph.cmid_weight = 0` *is* GraphDINO. **`ssl_neuron/eyewire2/00_giclmorph_spec.md` is the authority for it** — especially §2: the paper's 3D-PCA projection is replaced on RGCs by depth-anchored views (`projection.retina_views`), because on a flat arbor PC1/PC2 lie in xy and the paper's views never show depth.

## Environment & packaging

- Packaged with `uv`/`pyproject.toml` (hatchling backend), not the legacy `setup.py` the README still mentions.
- Base install (`uv sync`) only needs `numpy`, `tqdm`, `seaborn`, `matplotlib`, `networkx` — enough to preprocess/visualize data on any machine (including Windows), no GPU or `torch` required.
- The `torch` optional-dependency group (`uv sync --extra torch`) adds `torch>=2.3`, `torchvision>=0.18`, `scikit-learn>=1.4`, `scipy` — required for anything that actually builds/trains/runs the model (`datasets.py`'s augmentation functions, `graphdino.py`, `train.py`).
- `ssl_neuron/utils.py` and `ssl_neuron/data/data_utils.py` import `torch`/`scipy`/`pandas` lazily inside individual functions specifically so the rest of the module (plotting, graph utilities used by preprocessing) stays importable without those deps.
- Allen-specific preprocessing (`extract_allen_data.ipynb`) additionally needs `allensdk` + `pandas`; deliberately *not* declared as a pyproject extra because allensdk's old pins (e.g. `matplotlib<3.5`) would drag those versions into the single shared `uv.lock` used by every extra. Install ad hoc: `uv pip install allensdk pandas`.
- No test suite, linter, or CI config exists in this repo.

## Running things

Train from scratch on the ABA dataset (needs the `torch` extra and a GPU):
```
python3 ssl_neuron/main.py --config=ssl_neuron/configs/config.json
```

Eyewire2 retina pipeline (`ssl_neuron/eyewire2/`, run in this numbered order) — these are jupytext "percent"-format `.py` files (open directly as notebooks in Jupyter, or run as plain scripts):
1. `01_preprocess_data.py` — converts skeliner's `.swc` skeletons into the `features.npy`/`neighbors.pkl` layout `GraphDataset` expects (pure functions live in `eyewire2/preprocessing.py`); torch-free, runs locally on Windows. Input paths come from the `paths` block of `eyewire2/config.json` (cluster first, local mirror second; the first that exists wins), and the cell selection from `preprocessing.cellclass` (`"RGC"`; `null` keeps every class, a local smoke test only). The train/val split is stratified by celltype, so `07` has k-NN queries for every type. Also writes a `cell_meta.csv` sidecar: absolute soma positions (for the later mosaic stage, unrecoverable afterwards) and celltype labels (for evaluation only — training is fully self-supervised).
2. `02_visualize_data.py` — sanity checks, plus the measurements that settle the spec's open items; also torch-free. Warns if the skeletons on disk still have soma-centered z, i.e. were written by a pre-spec version of `01`.
3. `03_train_graphdino.py` — actually trains GraphDINO, via `RetinaGraphDataset`; needs the `torch` extra and a GPU, run on the cluster only.
4. `04_visualize_results.py` — loads the latest checkpoint and inspects the embedding.
5. `05_render_projections.py` — GICLMorph only: renders each cell's canonical 2D views into `skeletons/<cell_id>/projections.npy`; torch-free.
6. `06_train_giclmorph.py` — trains GICLMorph (`config_giclmorph.json`, `GICLTrainer`); GPU. With `cmid_weight: 0` it is the like-for-like GraphDINO control.
7. `07_evaluate_embeddings.py` — k-NN (also on consensus-labeled queries alone) / precision@k / linear-probe / clustering scores on celltype for every GraphDINO and GICLMorph run found, plus a no-learning depth-profile baseline; per-class recall, confusion matrices and example retrievals.

`05`–`07` read the `GICLMORPH_CONFIG`, `GICLMORPH_DATA`, `GICLMORPH_CKPTS` environment variables to override their paths.

On Windows, running `03`/`06`/`07` as plain scripts hangs in `GraphDataset.__init__`: its `multiprocessing.Manager()` is started under the `spawn` start method, the manager child fails to bootstrap from the unguarded script, and the parent waits forever. The scripts are meant for the Linux cluster (`fork`) or Jupyter, where this does not happen.

Checkpoints are written to the directory in `config['trainer']['ckpt_dir']`.

## Architecture

**Data pipeline** (`ssl_neuron/datasets.py`, `GraphDataset`): loads each cell's `features.npy` (node xyz, N x 3+) and `neighbors.pkl` (dict: node id -> set of neighbor ids) from `<data.path>/skeletons/<cell_id>/`, keyed by `train_ids.npy`/`val_ids.npy`. Assumes soma is node 0, node positions are in microns, y-axis orthogonal to pia, and axons already removed. On `__getitem__`, produces **two independently augmented views** of the same graph — this pair is the self-supervised signal DINO trains on. Augmentation = random subgraph deletion (`drop_random_branch`, `n_drop_branch` times) + subsampling down to `n_nodes` (`subsample_graph`, which iteratively removes degree-<3 non-protected nodes) + 3D rotation/jitter/translation of node positions (`ssl_neuron/utils.py`). The soma node is always protected from removal.

**Eyewire2 retina adaptation** (`ssl_neuron/eyewire2/`): `dataset.py`'s `RetinaGraphDataset` subclasses `GraphDataset` and overrides *only* `_augment_node_position`, swapping in `augment.py` (pure numpy, importable without `torch`). Two things differ from the ABA assumptions and will silently ruin a run if ignored:
- **Skeletons are soma-centered in x and y only.** z stays in the shared warped IPL-depth frame, because stratification depth is the main celltype signal and is only meaningful relative to the IPL. So no z-translation (`augment_positions` raises on one), no z-flip, no z-scaling.
- **`rotate_graph(axis='z')` in `utils.py` is not a rotation.** It zeroes the z row/column of a random 3D rotation; the remaining 2x2 xy block is not orthogonal, so it shears and rescales the arbor (median area factor 0.50). Fine-ish for ABA, fatal here where arbor size is a feature. The retina path uses a proper SO(2) rotation instead; the stock function is untouched and still used by ABA.

`datasets.py` was made config-driven for this: `jitter_var`/`rotation_axis`/`translate_var` are now optional (the retina config replaces them with an `augment` block), the load-time node cap is the `cache_nodes` key instead of a hardcoded 1000, and `build_dataloader` takes a `dataset_cls`. All defaults preserve the ABA behavior.

Unresolved decisions are tracked in section 7 of the spec — notably whether the warped z is in microns or percent-IPL-depth, which sets the z-jitter magnitude. `02_visualize_data.py` prints what is needed to settle them.

**Preprocessing utilities** (`ssl_neuron/utils.py`, `ssl_neuron/data/data_utils.py`): graph algorithms shared by both preprocessing (building `features.npy`/`neighbors.pkl` from raw skeletons) and augmentation (during training) — `neighbors_to_adjacency`/`neighbors_to_adjacency_torch`, `remap_neighbors` (reindex node ids to `0..N`), `connect_graph` (stitch disconnected components by nearest-node distance), `remove_axon` (strip axon-typed nodes using the one-hot compartment encoding in feature columns 4-7), `get_leaf_branch_nodes`/`compute_node_distances`/`drop_random_branch`/`traverse_dir` (branch-level graph surgery), `compute_eig_lapl_torch_batch` (graph Laplacian eigenvector positional encoding, batched).

**Model** (`ssl_neuron/graphdino.py`):
- `GraphAttention` — attention that interpolates between global transformer attention and local 1-hop message passing, via a per-node, per-head learned trade-off weight `gamma` (predicted from node features, exponentiated) applied to the raw attention adjacency.
- `GraphTransformer` — stack of `AttentionBlock`s; prepends a CLS token (with its own learned positional embedding) to the node sequence; node features go through `to_node_embedding`, Laplacian positional encodings through `to_pos_embedding`, both added before the blocks. Returns both a raw CLS embedding (`mlp_head`) and a projected embedding (`projector`, `proj_dim` -> `num_classes`) used for the DINO loss.
- `GraphDINO` — wraps a student/teacher pair of identical `GraphTransformer`s (teacher = EMA of student, `update_moving_average`, no gradients). Loss is cross-view self-distillation: student's view-2 projection vs. teacher's view-1 projection (softmax with separate student/teacher temperatures, teacher additionally centered by a running-average `teacher_centers`), symmetrized over both view orderings.
- `create_model(config)` is the single entry point that wires config -> `GraphTransformer` -> `GraphDINO`.

**Training loop** (`ssl_neuron/train.py`, `Trainer`): iteration-based (not purely epoch-based) — `max_iter` counts dataloader steps across epochs, with linear LR warmup (`max_iter // 50` steps) then exponential decay (halving every `max_iter // 5` steps by default). Each step: compute Laplacian positional encodings for both augmented adjacency matrices, forward through `GraphDINO`, backprop, then `model.update_moving_average()` to EMA-update the teacher. Checkpoints saved every `save_ckpt_every` epochs as `ckpt_<epoch>.pt` (full `state_dict`, not just weights).

**Config-driven, not CLI-flag-driven**: nearly everything (model dims, augmentation strengths, `n_nodes`, optimizer schedule, checkpoint dir) flows through a single JSON config (`ssl_neuron/configs/config.json` for ABA, `ssl_neuron/eyewire2/config.json` for the retina pipeline) passed by path to `main.py`/loaded directly in the eyewire2 scripts, not overridden via argparse flags.

## Custom dataset conventions (see `ssl_neuron/data/README.md`)

To point `GraphDataset` at a new dataset, the data directory needs:
- `skeletons/<cell_id>/features.npy` (N x feat_dim, node features) and `skeletons/<cell_id>/neighbors.pkl` (dict of node id -> neighbor ids) per cell.
- `train_ids.npy` / `val_ids.npy` listing cell ids for each split.
- Soma node must be index 0; axons must already be removed; graph must be a single connected component.

The eyewire2 preprocessing script (`ssl_neuron/eyewire2/01_preprocess_data.py`) is the reference implementation of building this layout from raw `.swc` skeletons.
