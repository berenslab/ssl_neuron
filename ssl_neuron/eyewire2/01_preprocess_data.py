# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: ssl_neuron
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preprocess eyewire2 retina skeletons for GraphDINO
#
# Converts the `.swc` skeletons produced by `skeliner` into the
# `features.npy` / `neighbors.pkl` layout that `ssl_neuron.datasets.GraphDataset`
# expects (see `ssl_neuron/data/README.md`), following the same steps as
# `ssl_neuron/data/extract_allen_data.ipynb`:
#
# 1. center each skeleton on its soma,
# 2. make sure the skeleton graph is a single connected component
#    (`ssl_neuron.data.data_utils.connect_graph`),
# 3. drop axon nodes so only soma + dendrites remain
#    (`ssl_neuron.data.data_utils.remove_axon`),
# 4. save `features.npy` (xyz only) + `neighbors.pkl` per cell, plus a
#    `train_ids.npy` / `val_ids.npy` split.
#
# This notebook only needs numpy/networkx/matplotlib — no `torch` required,
# so it runs fine locally on Windows. Re-run it on the cluster (pointed at
# the cluster's larger skeleton directory) before training there.

# %%
import pickle
from pathlib import Path

import numpy as np
import networkx as nx
from tqdm import tqdm

from ssl_neuron.data.data_utils import connect_graph, remove_axon
from ssl_neuron.utils import neighbors_to_adjacency, plot_neuron


def _is_connected(neighbors):
    """ Cheap O(n) connectivity check via a sparse graph. `connect_graph`
    itself needs a dense N x N adjacency matrix, which is far too slow/large
    to build just to *check* connectivity on skeletons with thousands of
    nodes -- so only build it (below) on the rare graph that actually needs
    fixing up. """
    graph = nx.Graph()
    graph.add_nodes_from(neighbors.keys())
    for i, neigh in neighbors.items():
        for j in neigh:
            graph.add_edge(i, j)
    return nx.number_connected_components(graph) == 1

# %% [markdown]
# #### Config

# %%
try:
    THIS_DIR = Path(__file__).resolve().parent
except NameError:
    THIS_DIR = Path.cwd()

# Adjust RAW_DIR for the machine this runs on (e.g. the cluster's own skeleton store).
RAW_DIR = THIS_DIR.parents[2] / "data" / "morphologies-ew2" / "skel_final"
OUT_DIR = THIS_DIR / "data"

VAL_FRACTION = 0.1
SEED = 0

assert RAW_DIR.is_dir(), f"Raw skeleton directory not found: {RAW_DIR}"


# %% [markdown]
# #### SWC parsing
#
# Standard 7-column SWC (`id type x y z radius parent`). Node type follows the
# usual convention (1 or -1 = soma, 2 = axon, 3 = dendrite, >=4 = apical
# dendrite); most of these retina skeletons only have soma + dendrite nodes,
# but a few also carry a reconstructed axon.

# %%
def load_swc(path):
    ids, types, xyz, radii, parents = [], [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            n_id, n_type, x, y, z, r, parent = line.split()
            ids.append(int(n_id))
            types.append(int(n_type))
            xyz.append((float(x), float(y), float(z)))
            radii.append(float(r))
            parents.append(int(parent))

    ids = np.array(ids)
    id2idx = {node_id: i for i, node_id in enumerate(ids)}
    xyz = np.array(xyz)
    radii = np.array(radii)
    types = np.array(types)
    parent_idx = np.array([id2idx[p] if p != -1 else -1 for p in parents])
    return xyz, radii, types, parent_idx


def type_to_onehot(types):
    """ One-hot encode SWC type as [soma, axon, dendrite, apical], matching
    the column layout `ssl_neuron.data.data_utils.remove_axon` expects. """
    onehot = np.zeros((len(types), 4))
    onehot[np.isin(types, [1, -1]), 0] = 1  # soma
    onehot[types == 2, 1] = 1  # axon
    onehot[types == 3, 2] = 1  # dendrite
    onehot[types >= 4, 3] = 1  # apical dendrite
    return onehot


def build_neighbors(parent_idx):
    neighbors = {i: set() for i in range(len(parent_idx))}
    for i, p in enumerate(parent_idx):
        if p != -1:
            neighbors[i].add(int(p))
            neighbors[int(p)].add(i)
    return neighbors


def move_to_front(idx, features, neighbors):
    """ Relabel node indices so that node `idx` becomes node 0.
    `GraphDataset` hardcodes the soma as node 0, so this must hold even if
    the SWC root isn't already the first row. """
    if idx == 0:
        return features, neighbors
    order = list(range(len(features)))
    order[0], order[idx] = order[idx], order[0]
    old2new = {old: new for new, old in enumerate(order)}
    new_features = features[order]
    new_neighbors = {old2new[k]: {old2new[v] for v in vs} for k, vs in neighbors.items()}
    return new_features, new_neighbors


# %% [markdown]
# #### Find skeletons
#
# Prefer the `_cal` variant of each skeleton when present (better radius
# estimates, otherwise identical) -- radius isn't used as a model feature,
# but it's kept around in case it's useful later (e.g. for filtering).

# %%
def find_swc_files(raw_dir):
    files = {}
    for path in sorted(raw_dir.rglob("*.swc")):
        if path.stem.endswith("_cal"):
            continue
        cell_id = path.stem
        cal_path = path.with_name(f"{cell_id}_cal.swc")
        files[cell_id] = cal_path if cal_path.exists() else path
    return files


swc_files = find_swc_files(RAW_DIR)
print(f"Found {len(swc_files)} skeletons under {RAW_DIR}")


# %% [markdown]
# #### Preprocess each cell

# %%
def preprocess_cell(swc_path):
    xyz, radii, types, parent_idx = load_swc(swc_path)
    onehot = type_to_onehot(types)
    features = np.concatenate([xyz, radii[:, None], onehot], axis=1)  # N x 8
    neighbors = build_neighbors(parent_idx)

    root_idx = int(np.where(parent_idx == -1)[0][0])
    features, neighbors = move_to_front(root_idx, features, neighbors)
    soma_id = 0

    # Center on soma.
    features[:, :3] -= features[soma_id, :3]

    # Ensure a single connected component (skeletons should already be
    # trees, but be defensive about stray disconnected fragments).
    if not _is_connected(neighbors):
        adj_matrix = neighbors_to_adjacency(neighbors, range(len(neighbors)))
        adj_matrix, neighbors = connect_graph(adj_matrix, neighbors, features)

    # Drop axon nodes, keeping only soma + dendrites.
    neighbors, features, soma_id = remove_axon(neighbors, features, soma_id)
    assert soma_id == 0
    assert len(features) == len(neighbors)

    return features[:, :3].astype(np.float32), neighbors


processed_ids = []
for cell_id, swc_path in tqdm(list(swc_files.items())):
    features, neighbors = preprocess_cell(swc_path)

    cell_dir = OUT_DIR / "skeletons" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    np.save(cell_dir / "features.npy", features)
    with open(cell_dir / "neighbors.pkl", "wb") as f:
        pickle.dump(neighbors, f, pickle.HIGHEST_PROTOCOL)

    processed_ids.append(cell_id)

print(f"Preprocessed {len(processed_ids)} / {len(swc_files)} skeletons -> {OUT_DIR}")

# %% [markdown]
# #### Sanity-check one cell

# %%
sample_id = processed_ids[0]
sample_features = np.load(OUT_DIR / "skeletons" / sample_id / "features.npy")
with open(OUT_DIR / "skeletons" / sample_id / "neighbors.pkl", "rb") as f:
    sample_neighbors = pickle.load(f)

plot_neuron(sample_neighbors, sample_features)

# %% [markdown]
# #### Train / val split

# %%
rng = np.random.default_rng(SEED)
ids = np.array(processed_ids)
rng.shuffle(ids)

n_val = max(1, int(round(len(ids) * VAL_FRACTION)))
val_ids, train_ids = ids[:n_val], ids[n_val:]

np.save(OUT_DIR / "train_ids.npy", train_ids)
np.save(OUT_DIR / "val_ids.npy", val_ids)

print(f"{len(train_ids)} train / {len(val_ids)} val skeletons written to {OUT_DIR}")
