# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SSL-Neuron / GraphDINO: self-supervised graph representation learning for neuronal morphologies (skeleton graphs), from Weis et al. 2023 ("Self-Supervised Graph Representation Learning for Neuronal Morphologies", TMLR). GraphDINO is a graph-transformer trained with a DINO-style student/teacher self-distillation objective on augmented views of the same neuron skeleton.

The `ssl_neuron/eyewire2/` subfolder adapts the original Allen Brain Atlas (ABA) pipeline to eyewire2 retina skeletons produced by the sibling `skeliner` repo.

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
1. `01_preprocess_data.py` — converts skeliner's `.swc` skeletons into the `features.npy`/`neighbors.pkl` layout `GraphDataset` expects; only needs numpy/networkx/matplotlib, no `torch`, so it runs locally on Windows. `RAW_DIR` is hardcoded relative to this file's location and must be adjusted per machine (e.g. pointed at the cluster's own skeleton store) before rerunning there.
2. `02_visualize_data.py` — sanity-checks preprocessed output (node-count distribution, per-axis extent to pick `rotation_axis`, example plots); also torch-free.
3. `03_train_graphdino.py` — actually trains GraphDINO; needs the `torch` extra and a GPU, run on the cluster only.

Checkpoints are written to the directory in `config['trainer']['ckpt_dir']`.

## Architecture

**Data pipeline** (`ssl_neuron/datasets.py`, `GraphDataset`): loads each cell's `features.npy` (node xyz, N x 3+) and `neighbors.pkl` (dict: node id -> set of neighbor ids) from `<data.path>/skeletons/<cell_id>/`, keyed by `train_ids.npy`/`val_ids.npy`. Assumes soma is node 0, node positions are in microns, y-axis orthogonal to pia, and axons already removed. On `__getitem__`, produces **two independently augmented views** of the same graph — this pair is the self-supervised signal DINO trains on. Augmentation = random subgraph deletion (`drop_random_branch`, `n_drop_branch` times) + subsampling down to `n_nodes` (`subsample_graph`, which iteratively removes degree-<3 non-protected nodes) + 3D rotation/jitter/translation of node positions (`ssl_neuron/utils.py`). The soma node is always protected from removal.

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
