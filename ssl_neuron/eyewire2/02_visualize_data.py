# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: ssl_neuron
#     language: python
#     name: ssl_neuron
# ---

# %% [markdown]
# # Visualize preprocessed eyewire2 retina skeletons
#
# Loads the `features.npy` / `neighbors.pkl` pairs written by
# `01_preprocess_data.py` directly (no `GraphDataset`/`torch` needed), so this
# runs locally on Windows without the `torch` extra installed.
#
# Besides the basic sanity checks, this notebook settles three open items from
# `00_dataset_spec.md`:
#
# * **§7.1 — what unit is z in?** The pooled z-distribution across cells shows
#   whether z is microns or percent-IPL-depth, and whether the depth frame is
#   really shared (IPL band structure should be visible across cells).
# * **§7.3 — is skeliner's node spacing uniform?** The model's depth profile is
#   a histogram of *nodes*; that only equals a dendritic *length* profile if
#   nodes are equidistant along the skeleton.
# * **§4 — do the augmentations behave?** Two augmented views of one cell,
#   plus a numeric check that the xy rotation is norm-preserving and leaves z
#   untouched.
#
# And, using the celltype labels in `cell_meta.csv` (evaluation only -- nothing
# here feeds back into training), two checks on what the embedding is supposed
# to pick up: whether celltypes stratify at *different* depths (the direct test
# of the shared-frame assumption in §1.2), and whether dendritic field size
# still spans several-fold across cells (§1.4).

# %%
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from ssl_neuron.utils import plot_neuron
from ssl_neuron.eyewire2.augment import augment_positions

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

# Celltype labels, for the label-dependent checks below. A missing sidecar
# means the skeletons predate the current `01` (and are stale, see §6.2).
META_PATH = DATA_DIR / "cell_meta.csv"
if META_PATH.exists():
    meta = pd.read_csv(META_PATH, index_col="cell_id", dtype={"cell_id": str})
    not_in_meta = set(cell_ids) - set(meta.index)
    if not_in_meta:
        print(f"{len(not_in_meta)} skeleton directories are not in cell_meta.csv "
              f"(left over from an earlier run of 01) -- skipped")
        cell_ids = [c for c in cell_ids if c in meta.index]
    valid = meta.get("valid_celltype_final", pd.Series(True, index=meta.index))
    celltype = meta["celltype_final"].where(valid.fillna(False).astype(bool)).reindex(cell_ids)
else:
    print(f"WARNING: no {META_PATH.name} -- celltype checks are skipped. "
          f"Re-run 01_preprocess_data.py.")
    meta = None
    celltype = pd.Series(np.nan, index=cell_ids, dtype=object)
print(f"{int(celltype.notna().sum())} of them carry a celltype label, "
      f"over {celltype.nunique()} types")


def load_cell(cell_id):
    features = np.load(DATA_DIR / "skeletons" / cell_id / "features.npy")
    with open(DATA_DIR / "skeletons" / cell_id / "neighbors.pkl", "rb") as f:
        neighbors = pickle.load(f)
    return features, neighbors


def edge_array(neighbors):
    """ Unique edges as an (E, 2) index array. """
    return np.array([(i, j) for i, nbrs in neighbors.items() for j in nbrs if j > i])


def plot_skeleton(ax, features, neighbors, ax1=0, ax2=1, color="tab:blue"):
    """ `plot_neuron`, but one LineCollection instead of one `plot` call per
    edge (fast enough for full-size skeletons), and without resetting the axis
    limits, so several cells can share them. """
    edges = edge_array(neighbors)
    segments = features[edges][:, :, [ax1, ax2]]
    ax.add_collection(LineCollection(segments, colors=color, linewidths=0.5))
    ax.scatter(*features[0, [ax1, ax2]], color="k", s=10, zorder=10)


# %% [markdown]
# #### Node-count distribution
#
# Sanity-check `config['data']['n_nodes']` (nodes per augmented view) and
# `cache_nodes` (nodes kept per cell at load time) against how big these
# skeletons actually are. Cells with fewer than `n_nodes` nodes are silently
# skipped by `GraphDataset`.

# %%
n_nodes_per_cell = np.array([len(np.load(DATA_DIR / "skeletons" / c / "features.npy"))
                             for c in cell_ids])

plt.hist(n_nodes_per_cell, bins=30)
plt.axvline(config["data"]["n_nodes"], color="k", ls="--", label="n_nodes")
plt.axvline(config["data"]["cache_nodes"], color="k", ls=":", label="cache_nodes")
plt.xlabel("# nodes")
plt.ylabel("# cells")
plt.legend()
plt.title("Skeleton size distribution")
plt.show()

print(f"min={n_nodes_per_cell.min()}, median={int(np.median(n_nodes_per_cell))}, "
      f"max={n_nodes_per_cell.max()}")
print(f"{int((n_nodes_per_cell < config['data']['n_nodes']).sum())} cells below n_nodes "
      f"(these would be dropped by GraphDataset)")

# %% [markdown]
# #### Per-axis extent
#
# z should be the thin axis (IPL depth); x and y are the dendritic field. Note
# that the z *extent* is per-cell, so it says nothing about where in the IPL a
# cell sits -- that comes next.

# %%
extents = np.array([np.ptp(np.load(DATA_DIR / "skeletons" / c / "features.npy"), axis=0)
                    for c in cell_ids])

print(f"Median extent per axis (x, y, z): {np.median(extents, axis=0).round(1)}")
print(f"p10: {np.percentile(extents, 10, axis=0).round(1)}, "
      f"p90: {np.percentile(extents, 90, axis=0).round(1)}")

# %% [markdown]
# #### §1.4 Dendritic field size must span several-fold
#
# Field size is a feature the model is meant to use (alpha types differ from
# small-field types mostly by size), so nothing in preprocessing may have
# rescaled cells to a common size -- the xy extent below must spread widely.
# A long left tail is the other thing to look for: large-but-clipped arbors at
# the volume boundary present as small cells and pass the node-count QC
# (§7.2).

# %%
xy_extent = extents[:, :2].max(axis=1)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(xy_extent, bins=40)
ax.set(xlabel="xy extent (max of x, y)", ylabel="# cells", title="Dendritic field size")
plt.show()

p10, p50, p90 = np.percentile(xy_extent, [10, 50, 90])
print(f"xy extent: p10={p10:.0f}, median={p50:.0f}, p90={p90:.0f} -- "
      f"a {p90 / p10:.1f}x spread between p10 and p90")

# %% [markdown]
# #### §7.1 Is the depth frame shared, and in what unit?
#
# x and y are soma-centered, z is not. So soma z should *vary* across cells
# (if it is 0 everywhere, the skeletons were written by the old preprocessing
# and must be regenerated), and the pooled node-z distribution should show
# structure at fixed depths -- the IPL bands -- rather than one smooth blob.
#
# The overall z range also settles the unit: a full-IPL span of ~40 means
# microns, a span of ~100 means percent IPL depth. `config['data']['augment']`
# jitter must be scaled to match.

# %%
soma_z = np.array([np.load(DATA_DIR / "skeletons" / c / "features.npy")[0, 2] for c in cell_ids])
all_z = np.concatenate([np.load(DATA_DIR / "skeletons" / c / "features.npy")[:, 2]
                        for c in cell_ids])

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].hist(soma_z, bins=30)
axes[0].set_xlabel("soma z")
axes[0].set_ylabel("# cells")
axes[0].set_title("Soma depth across cells")

axes[1].hist(all_z, bins=100)
axes[1].set_xlabel("node z")
axes[1].set_ylabel("# nodes")
axes[1].set_title("Pooled depth distribution (all cells)")
plt.tight_layout()
plt.show()

print(f"soma z: min={soma_z.min():.2f}, median={np.median(soma_z):.2f}, max={soma_z.max():.2f}")
print(f"node z: {np.percentile(all_z, 1):.2f} .. {np.percentile(all_z, 99):.2f} (1st-99th pct), "
      f"full range {all_z.min():.2f} .. {all_z.max():.2f}")
if np.allclose(soma_z, 0):
    print("WARNING: every soma sits at z=0 -- these skeletons are soma-centered in z. "
          "Re-run 01_preprocess_data.py (see 00_dataset_spec.md section 6.2).")

# %% [markdown]
# #### §7.1 Do celltypes stratify at different depths?
#
# The direct test of the shared-frame assumption. Each curve is one celltype's
# mean stratification profile (each cell's node-z histogram, normalized, then
# averaged so every cell counts once). If the depth frame is shared, types peak
# at clearly different depths; if the curves lie on top of each other, z is not
# comparable across cells and no amount of training will separate the types on
# depth.

# %%
N_PROFILE_CLASSES = 8
MIN_PROFILE_CELLS = 3

type_counts = celltype.value_counts()
profile_types = type_counts[type_counts >= MIN_PROFILE_CELLS].index[:N_PROFILE_CLASSES]
depth_bins = np.linspace(np.percentile(all_z, 0.5), np.percentile(all_z, 99.5), 61)
depth_centers = 0.5 * (depth_bins[1:] + depth_bins[:-1])

def depth_profile(cell_id, bins):
    z = np.load(DATA_DIR / "skeletons" / cell_id / "features.npy")[:, 2]
    return np.histogram(z, bins=bins)[0] / len(z)


if len(profile_types) == 0:
    print(f"No celltype has {MIN_PROFILE_CELLS}+ cells here -- skipped.")
else:
    fig, ax = plt.subplots(figsize=(7, 4))
    for t in profile_types:
        members = celltype.index[celltype == t]
        profile = np.mean([depth_profile(c, depth_bins) for c in members], axis=0)
        ax.plot(depth_centers, profile, label=f"{t} (n={len(members)})")
    ax.set(xlabel="node z", ylabel="fraction of nodes",
           title=f"Stratification profiles, {len(profile_types)} largest celltypes")
    ax.legend(fontsize="x-small")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# #### §7.3 Is node spacing uniform along the skeleton?
#
# The depth profile the model sees is the z-histogram of its sampled *nodes*.
# That is a dendritic *length* density profile only if nodes are equidistant
# along the skeleton. If the edge-length distribution below is tight, keeping
# skeliner's native spacing is fine; if it is broad or bimodal, the profile is
# reweighted by whatever drives the spacing, and arc-length resampling in
# preprocessing is worth revisiting.

# %%
N_SPACING_CELLS = min(20, len(cell_ids))
spacing_ids = list(np.random.default_rng(0).choice(cell_ids, N_SPACING_CELLS, replace=False))

lengths = []
for cell_id in spacing_ids:
    features, neighbors = load_cell(cell_id)
    edges = edge_array(neighbors)
    lengths.append(np.linalg.norm(features[edges[:, 0]] - features[edges[:, 1]], axis=1))
lengths = np.concatenate(lengths)

plt.hist(lengths, bins=100)
plt.xlabel("edge length")
plt.ylabel("# edges")
plt.title(f"Node spacing over {N_SPACING_CELLS} cells")
plt.show()

print(f"edge length: median={np.median(lengths):.3f}, "
      f"IQR={np.percentile(lengths, 25):.3f}..{np.percentile(lengths, 75):.3f}, "
      f"max={lengths.max():.3f}")
print(f"coefficient of variation = {lengths.std() / lengths.mean():.2f}")
print("Spread alone does not settle the question -- what matters is whether the "
      "spacing varies *with depth*. That is measured two cells down.")

# %% [markdown]
# #### Depth profiles: what the model sees vs. the classical feature
#
# Solid = dendritic length per depth bin (each edge contributes its length at
# its midpoint), i.e. the classical z-density profile. Dashed = plain node
# histogram, which is what GraphDINO's subsampled views approximate. The two
# should have the same shape; where they don't, node spacing is the culprit.

# %%
def z_profiles(features, neighbors, bins):
    edges = edge_array(neighbors)
    seg_len = np.linalg.norm(features[edges[:, 0]] - features[edges[:, 1]], axis=1)
    seg_z = features[edges, 2].mean(axis=1)
    by_length, _ = np.histogram(seg_z, bins=bins, weights=seg_len)
    by_node, _ = np.histogram(features[:, 2], bins=bins)
    return by_length / max(by_length.sum(), 1), by_node / max(by_node.sum(), 1)


bins = np.linspace(np.percentile(all_z, 0.5), np.percentile(all_z, 99.5), 41)
centers = 0.5 * (bins[1:] + bins[:-1])

n_examples = min(4, len(cell_ids))
fig, axes = plt.subplots(1, n_examples, figsize=(3.2 * n_examples, 3), sharex=True)
for ax, cell_id in zip(np.atleast_1d(axes), cell_ids[:n_examples]):
    features, neighbors = load_cell(cell_id)
    by_length, by_node = z_profiles(features, neighbors, bins)
    ax.plot(centers, by_length, label="by length")
    ax.plot(centers, by_node, ls="--", label="by node")
    ax.set_title(cell_id[-6:])
    ax.set_xlabel("z")
np.atleast_1d(axes)[0].set_ylabel("fraction")
np.atleast_1d(axes)[0].legend(fontsize="small")
plt.tight_layout()
plt.show()

# %% [markdown]
# The verdict on §7.3, over every cell: how far apart are the two profiles?
# Uneven node spacing only distorts the depth profile if the spacing varies
# *with depth*, so this -- not the spread of edge lengths -- is the number that
# decides whether arc-length resampling is needed. A correlation above ~0.99
# and an L1 distance of a few percent means the node histogram the model sees
# is a faithful stand-in for dendritic length density.

# %%
l1_distance, correlation = [], []
for cell_id in cell_ids:
    features, neighbors = load_cell(cell_id)
    by_length, by_node = z_profiles(features, neighbors, bins)
    l1_distance.append(np.abs(by_length - by_node).sum())
    correlation.append(np.corrcoef(by_length, by_node)[0, 1])

l1_distance, correlation = np.array(l1_distance), np.array(correlation)
print(f"length- vs node-weighted depth profile over {len(cell_ids)} cells:")
print(f"  L1 distance:  median {np.median(l1_distance):.3f}, "
      f"p90 {np.percentile(l1_distance, 90):.3f}, max {l1_distance.max():.3f}  (0 = identical)")
print(f"  correlation:  median {np.median(correlation):.4f}, min {correlation.min():.4f}")

# %% [markdown]
# #### Example skeletons
#
# Random cells, all drawn on the same axis limits so relative field size and
# depth are visible at a glance. Top row: xy (dendritic field, soma-centered).
# Bottom row: xz (stratification depth, *not* centered -- cells sit at
# different depths on purpose).

# %%
N_SHOW = min(5, len(cell_ids))
show_ids = list(np.random.default_rng(1).choice(cell_ids, N_SHOW, replace=False))
show_cells = [load_cell(c) for c in show_ids]
xy_lim = max(np.abs(f[:, :2]).max() for f, _ in show_cells) * 1.05
z_lim = (all_z.min() - 1, all_z.max() + 1)

fig, axes = plt.subplots(2, N_SHOW, figsize=(3 * N_SHOW, 6.5),
                         gridspec_kw={"height_ratios": [3, 1]})
axes = axes.reshape(2, N_SHOW)
for col, (cell_id, (features, neighbors)) in enumerate(zip(show_ids, show_cells)):
    plot_skeleton(axes[0, col], features, neighbors, 0, 1)
    plot_skeleton(axes[1, col], features, neighbors, 0, 2)
    axes[0, col].set(xlim=(-xy_lim, xy_lim), ylim=(-xy_lim, xy_lim), aspect="equal")
    axes[1, col].set(xlim=(-xy_lim, xy_lim), ylim=z_lim)
    label = celltype.get(cell_id)
    axes[0, col].set_title(f"{label if isinstance(label, str) else 'untyped'}\n"
                           f"{cell_id} ({len(features)} nodes)", fontsize="x-small")

axes[0, 0].set_ylabel("y")
axes[1, 0].set_ylabel("z (depth)")
plt.tight_layout()
plt.show()

# %% [markdown]
# #### §4 Augmentation preview
#
# Two independently augmented views of the same cell, using the `augment`
# block from `config.json`. In xy the arbor should rotate (and sometimes
# mirror) rigidly; in xz it should stay put in depth.

# %%
aug_kwargs = config["data"]["augment"]
features, neighbors = load_cell(cell_ids[0])

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for col, title in enumerate(["original", "view 1", "view 2"]):
    pos = features if col == 0 else augment_positions(features, **aug_kwargs)
    plot_neuron(neighbors, pos, ax1=0, ax2=1, ax=axes[0, col])
    plot_neuron(neighbors, pos, ax1=0, ax2=2, ax=axes[1, col])
    axes[0, col].set_title(title)

axes[0, 0].set_ylabel("y")
axes[1, 0].set_ylabel("z (depth)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Numeric check of the same thing: with scaling and noise switched off, the
# rotation must preserve every pairwise xy distance exactly and leave z
# untouched. (The stock `ssl_neuron.utils.rotate_graph(axis='z')` fails this --
# see spec §6.1.)

# %%
rigid = augment_positions(features, rotate_xy=True, mirror_xy=True,
                          scale_xy=0.0, jitter=(0, 0, 0), translate=(0, 0, 0))

sample = np.random.default_rng(0).choice(len(features), size=min(500, len(features)), replace=False)
before = np.linalg.norm(features[sample, None, :2] - features[None, sample, :2], axis=-1)
after = np.linalg.norm(rigid[sample, None, :2] - rigid[None, sample, :2], axis=-1)

print(f"max relative change in pairwise xy distance: "
      f"{np.abs(after - before).max() / max(before.max(), 1e-9):.2e}")
print(f"max change in z: {np.abs(rigid[:, 2] - features[:, 2]).max():.2e}")

# %% [markdown]
# #### The evaluation subset: class balance and label provenance
#
# Training ignores labels, but `07_evaluate_embeddings.py` scores on them, and
# two properties of this subset shape what its numbers mean:
#
# * **Balance.** The classes are strongly imbalanced, which is why `07` reports
#   *balanced* accuracy. Its k-NN only scores classes with at least
#   `evaluation.min_cells_per_class` labeled cells in `train_ids`.
# * **Provenance** (`celltype_final_decision`). `classifier` labels were
#   assigned by a model rather than by human consensus (`both_strong` /
#   `both_weak`), so a k-NN score over them partly measures agreement with that
#   classifier. `07` therefore also scores the consensus-labeled queries alone.

# %%
with open(THIS_DIR / "config_giclmorph.json") as f:
    MIN_CELLS = json.load(f)["evaluation"]["min_cells_per_class"]

if celltype.notna().sum() == 0:
    print("No labeled cells -- skipped.")
else:
    decision = (meta["celltype_final_decision"].reindex(celltype.index).fillna("unknown")
                if "celltype_final_decision" in meta.columns
                else pd.Series("unknown", index=celltype.index))
    table = pd.crosstab(celltype.dropna(), decision[celltype.notna()])
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(6, 0.25 * len(table)), 5))
    bottom = np.zeros(len(table))
    for col in table.columns:
        ax.bar(range(len(table)), table[col], bottom=bottom, label=col)
        bottom += table[col].to_numpy()
    ax.set_xticks(range(len(table)), table.index, rotation=90, fontsize="x-small")
    ax.set_ylabel("# cells")
    ax.set_title("celltype_final over the preprocessed cells")
    ax.legend(title="decision", fontsize="x-small")
    plt.tight_layout()
    plt.show()

    train_ids = {str(c) for c in np.load(DATA_DIR / "train_ids.npy")}
    train_counts = celltype[celltype.index.isin(train_ids)].value_counts()
    kept = train_counts[train_counts >= MIN_CELLS]
    print(f"{len(table)} celltypes, {int(table.to_numpy().sum())} labeled cells; "
          f"largest {table.sum(axis=1).max()}, smallest {table.sum(axis=1).min()}")
    print(f"07 will score {len(kept)} classes ({int(kept.sum())} train cells with "
          f">= {MIN_CELLS} per class)")
    print("label provenance: " + ", ".join(f"{k} {v}" for k, v in table.sum().items()))

# %%
