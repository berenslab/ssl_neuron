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
# # Visualize preprocessed eyewire2 retina skeletons
#
# Loads the `features.npy` / `neighbors.pkl` pairs written by
# `01_preprocess_data.py` directly (no `GraphDataset`/`torch` needed), so
# this also runs locally on Windows without the `torch` extra installed.
#
# `GraphDataset`-based augmented-pair visualization (like in
# `ssl_neuron/demos/load_data.ipynb`) needs `torch`; run that on the cluster
# instead once the `torch` extra is installed.

# %%
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ssl_neuron.utils import plot_neuron

# %% [markdown]
# #### Config

# %%
try:
    THIS_DIR = Path(__file__).resolve().parent
except NameError:
    THIS_DIR = Path.cwd()

DATA_DIR = THIS_DIR / "data"

with open(THIS_DIR / "config.json") as f:
    config = json.load(f)

cell_ids = sorted(p.name for p in (DATA_DIR / "skeletons").iterdir() if p.is_dir())
print(f"{len(cell_ids)} preprocessed skeletons in {DATA_DIR}")

# %% [markdown]
# #### Node-count distribution
#
# Useful for sanity-checking `config['data']['n_nodes']` against how big
# these skeletons actually are (printed below).

# %%
n_nodes_per_cell = []
for cell_id in cell_ids:
    features = np.load(DATA_DIR / "skeletons" / cell_id / "features.npy")
    n_nodes_per_cell.append(len(features))

plt.hist(n_nodes_per_cell, bins=30)
plt.xlabel("# nodes")
plt.ylabel("# cells")
plt.title("Skeleton size distribution")
plt.show()

print(f"min={min(n_nodes_per_cell)}, median={int(np.median(n_nodes_per_cell))}, "
      f"max={max(n_nodes_per_cell)}, config n_nodes={config['data']['n_nodes']}")

# %% [markdown]
# #### Per-axis extent
#
# `GraphDataset` rotates each skeleton around a fixed `rotation_axis`,
# assumed to be the "flat"/stratification axis (orthogonal to the pia for
# cortical data). For retina skeletons that's the depth axis of the
# skeletonized volume rather than necessarily "y" -- check which axis has
# the smallest spread below and set `config.json`'s `data.rotation_axis`
# (x/y/z) accordingly before training.

# %%
extents = []
for cell_id in cell_ids:
    features = np.load(DATA_DIR / "skeletons" / cell_id / "features.npy")
    extents.append(features.max(axis=0) - features.min(axis=0))

median_extent = np.median(extents, axis=0)
print(f"Median extent per axis (x, y, z): {median_extent}")
print(f"config.json currently uses rotation_axis={config['data']['rotation_axis']!r}")

# %% [markdown]
# #### Plot a few example skeletons

# %%
n_examples = min(4, len(cell_ids))
fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 4), sharex=True, sharey=True)
if n_examples == 1:
    axes = [axes]

for ax, cell_id in zip(axes, cell_ids[:n_examples]):
    features = np.load(DATA_DIR / "skeletons" / cell_id / "features.npy")
    with open(DATA_DIR / "skeletons" / cell_id / "neighbors.pkl", "rb") as f:
        neighbors = pickle.load(f)

    plot_neuron(neighbors, features, ax=ax)
    ax.set_title(f"{cell_id}\n({len(features)} nodes)")

plt.tight_layout()
plt.show()
