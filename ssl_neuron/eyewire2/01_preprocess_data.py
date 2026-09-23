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
# # Preprocess eyewire2 retina skeletons for GraphDINO
#
# Converts the `.swc` skeletons produced by `skeliner` into the
# `features.npy` / `neighbors.pkl` layout that
# `ssl_neuron.eyewire2.dataset.RetinaGraphDataset` expects (see
# `ssl_neuron/data/README.md` for the layout, and `00_dataset_spec.md` in this
# folder for why this pipeline differs from the Allen one).
#
# Per cell (`ssl_neuron.eyewire2.preprocessing.preprocess_cell`):
#
# 1. parse the SWC into node features + a neighbor dict, soma first,
# 2. make sure the skeleton is a single connected component,
# 3. drop axon nodes so only soma + dendrites remain, then check connectivity
#    *again* -- removing mislabeled axon nodes can split the arbor,
# 4. QC on node count,
# 5. center **x and y only** on the soma; z stays in the shared warped
#    IPL-depth frame (spec sections 1.2 and 6.2),
# 6. save `features.npy` (xyz) + `neighbors.pkl`.
#
# Plus a `cell_meta.csv` sidecar carrying each cell's absolute soma position
# (for the later mosaic stage) and its celltype label (for evaluation).
#
# Only needs numpy/pandas/matplotlib -- no `torch` -- so it runs locally on
# Windows. Re-run it on the cluster (it picks the cluster paths from
# `config.json` automatically) before training there.
#
# > **Note:** skeletons written before the xy-only centering change have
# > soma-centered z and must be regenerated.

# %%
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from ssl_neuron.utils import plot_neuron
from ssl_neuron.eyewire2.preprocessing import MIN_NODES, SkeletonQCError, preprocess_cell

# %% [markdown]
# #### Config
#
# `config.json` lists the cluster path first and the local one second for both
# inputs; the first that exists wins, so the same file works in both places.
# `preprocessing.cellclass` selects the cells (spec section 3): `"RGC"` for real
# runs, `null` for every cellclass -- only useful as a local smoke test, since
# the ~110 skeletons mirrored outside the cluster are mostly amacrine cells.

# %%
try:
    THIS_DIR = Path(__file__).resolve().parent
except NameError:
    THIS_DIR = Path.cwd()

OUT_DIR = THIS_DIR / "data"

with open(THIS_DIR / "config.json") as f:
    config = json.load(f)


def first_existing(candidates, what):
    for candidate in candidates:
        if Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(f"None of the configured {what} paths exist: {candidates}")


SWC_DIR = first_existing(config["paths"]["swc_dir"], "swc_dir")
DF_PATH = first_existing(config["paths"]["dataframe"], "dataframe")
CELLCLASS = config["preprocessing"]["cellclass"]
print(f"skeletons:  {SWC_DIR}")
print(f"dataframe:  {DF_PATH}")

VAL_FRACTION = 0.1
SEED = 0

# Metadata columns, as produced by eyewire2-datajoint's `prepare_manuscript_df.py`.
# `cellclass_final` selects the training set; the `celltype_final` columns are
# carried through for evaluation only -- training is fully self-supervised.
META_COLUMNS = [
    "cellclass_final", "valid_cellclass_final", "valid_status",
    "celltype_final", "valid_celltype_final", "celltype_final_decision",
    "soma_x_um", "soma_y_um", "soma_z_um",
]


def load_metadata(path, columns=META_COLUMNS):
    """ Load cell metadata (the same dataframe `plot_RGCs.ipynb` uses). Falls
    back to a full read if this dataframe version is missing some column, so a
    renamed column shows up as a loud message instead of a crash. """
    try:
        return pd.read_parquet(path, columns=columns)
    except (KeyError, ValueError) as err:
        print(f"Column-limited read failed ({err}); falling back to a full read.")
        df = pd.read_parquet(path)
        missing = [c for c in columns if c not in df.columns]
        if missing:
            print(f"Missing columns in this dataframe version: {missing}")
        return df[[c for c in columns if c in df.columns]]


def select_training_cells(df, cellclass="RGC"):
    """ The SSL training set: every confidently-labeled cell of `cellclass`,
    typed or not (spec section 3). Labels are not used for training, so there is
    no reason to restrict this to cells that already have a celltype.
    `cellclass=None` keeps every class. """
    selected = pd.Series(True, index=df.index)
    if cellclass is not None:
        selected &= df["cellclass_final"] == cellclass
    for flag in ("valid_cellclass_final", "valid_status"):
        if flag in df.columns:
            selected &= df[flag].fillna(False).astype(bool)
    return df[selected]


df_meta = select_training_cells(load_metadata(DF_PATH), CELLCLASS)
cell_id_filter = {str(i) for i in df_meta.index}
print(f"{len(cell_id_filter)} {CELLCLASS or 'cells of any class'} selected from {DF_PATH.name}")
if "celltype_final" in df_meta.columns:
    n_typed = int(df_meta["celltype_final"].notna().sum())
    print(f"  of which {n_typed} carry a celltype label (evaluation subset)")

# %% [markdown]
# #### Find skeletons
#
# Prefer the `_cal` variant of each skeleton when present (better radius
# estimates, otherwise identical) -- radius isn't used as a model feature,
# but it's kept around in case it's useful later (e.g. for filtering).

# %%
def find_swc_files(swc_dir):
    files = {}
    for path in sorted(swc_dir.rglob("*.swc")):
        if path.stem.endswith("_cal"):
            continue
        cell_id = path.stem
        cal_path = path.with_name(f"{cell_id}_cal.swc")
        files[cell_id] = cal_path if cal_path.exists() else path
    return files


swc_files = find_swc_files(SWC_DIR)
swc_files = {cell_id: path for cell_id, path in swc_files.items() if cell_id in cell_id_filter}
print(f"Found {len(swc_files)} / {len(cell_id_filter)} selected skeletons under {SWC_DIR}")
assert swc_files, "None of the selected cells have a skeleton under SWC_DIR"

# %% [markdown]
# #### Preprocess each cell
#
# Cells that fail QC are skipped and tallied by reason rather than silently
# dropped -- a truncated arbor is worse than no cell at all now that arbor size
# is a feature the model is meant to use.

# %%
rows = []
qc_failures = Counter()

for cell_id, swc_path in tqdm(list(swc_files.items())):
    try:
        features, neighbors, soma_xyz, info = preprocess_cell(swc_path, min_nodes=MIN_NODES)
    except SkeletonQCError as err:
        # Group by reason, not by the exact numbers in the message.
        qc_failures[re.sub(r"\d+", "N", str(err))] += 1
        continue

    cell_dir = OUT_DIR / "skeletons" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    np.save(cell_dir / "features.npy", features)
    with open(cell_dir / "neighbors.pkl", "wb") as f:
        pickle.dump(neighbors, f, pickle.HIGHEST_PROTOCOL)

    rows.append({
        "cell_id": cell_id,
        "soma_x": soma_xyz[0], "soma_y": soma_xyz[1], "soma_z": soma_xyz[2],
        **info,
    })

print(f"Preprocessed {len(rows)} / {len(swc_files)} skeletons -> {OUT_DIR}")
for reason, count in qc_failures.most_common():
    print(f"  dropped {count:5d}: {reason}")

# %% [markdown]
# #### Metadata sidecar
#
# One row per preprocessed cell: the absolute soma position that was subtracted
# during centering (the later mosaic stage needs it, and it is unrecoverable
# from `features.npy`), the node counts, and the celltype label used for
# evaluation.

# %%
df_cells = pd.DataFrame(rows).set_index("cell_id")
df_cells = df_cells.join(df_meta.rename(index=str), how="left")
df_cells.to_csv(OUT_DIR / "cell_meta.csv")

print(f"{len(df_cells)} cells written to {OUT_DIR / 'cell_meta.csv'}")
print(df_cells[["n_nodes", "n_axon_nodes", "n_stitched_edges"]].describe().round(1))

# %% [markdown]
# #### Check for stale skeletons
#
# Nothing is deleted automatically. Left-over directories from an earlier run
# (e.g. one written before the xy-only centering fix) would not be referenced
# by the new `train_ids.npy`, but they are confusing -- and dangerous if an old
# split file is still lying around.

# %%
on_disk = {p.name for p in (OUT_DIR / "skeletons").iterdir() if p.is_dir()}
stale = on_disk - set(df_cells.index)
if stale:
    print(f"WARNING: {len(stale)} skeleton directories in {OUT_DIR / 'skeletons'} "
          f"were not written by this run. Delete them if they are from an "
          f"older preprocessing version, e.g.:")
    print("  import shutil; [shutil.rmtree(OUT_DIR / 'skeletons' / c) for c in stale]")
else:
    print("No stale skeleton directories.")

# %% [markdown]
# #### Sanity-check one cell
#
# The soma should sit at x = y = 0, while z should *not* be centered: the z
# range below is the cell's stratification depth in the shared IPL frame.

# %%
sample_id = df_cells.index[0]
sample_features = np.load(OUT_DIR / "skeletons" / sample_id / "features.npy")
with open(OUT_DIR / "skeletons" / sample_id / "neighbors.pkl", "rb") as f:
    sample_neighbors = pickle.load(f)

print(f"{sample_id}: soma at {sample_features[0].round(2)}, "
      f"z range {sample_features[:, 2].min():.1f} .. {sample_features[:, 2].max():.1f}")
plot_neuron(sample_neighbors, sample_features)
plt.show()

# %% [markdown]
# #### Train / val split
#
# For training, the validation set only produces a DINO loss curve (there are
# no labels in the objective). But `07_evaluate_embeddings.py` uses the labeled
# cells of `val_ids` as its k-NN queries, and with ~28 celltypes a plain random
# 10% leaves many classes with 0-3 queries (`00_giclmorph_spec.md` section
# 7.6). So the split is stratified by celltype, with the unlabeled cells as one
# more stratum: every class gets its `VAL_FRACTION` share of val cells, and the
# unlabeled cells are split as a random split would.

# %%
def stratified_split(strata, val_fraction, seed):
    """ Systematic stratified sample: each stratum contributes its
    `val_fraction` share of val cells, rounded up or down at random, while the
    total stays exactly `round(N * val_fraction)` (at least 1). Plain per-stratum
    rounding would instead give every stratum below 1 / val_fraction cells no
    val cell at all. """
    rng = np.random.default_rng(seed)
    ids = strata.index.to_numpy().astype(str)
    key = np.empty(len(ids))
    for positions in strata.groupby(strata).indices.values():
        positions = rng.permutation(positions)
        key[positions] = (np.arange(len(positions)) + rng.random()) / len(positions)
    n_val = max(1, int(round(len(ids) * val_fraction)))
    order = np.argsort(key, kind="stable")
    return np.sort(ids[order[n_val:]]), np.sort(ids[order[:n_val]])


celltype = df_cells["celltype_final"].where(
    df_cells.get("valid_celltype_final", pd.Series(True, index=df_cells.index))
    .fillna(False).astype(bool))
train_ids, val_ids = stratified_split(celltype.fillna("<unlabeled>"), VAL_FRACTION, SEED)

np.save(OUT_DIR / "train_ids.npy", train_ids)
np.save(OUT_DIR / "val_ids.npy", val_ids)

print(f"{len(train_ids)} train / {len(val_ids)} val skeletons written to {OUT_DIR}")
n_val_labeled = int(celltype.reindex(val_ids).notna().sum())
print(f"  {n_val_labeled} of the val cells are labeled (k-NN queries in 07), "
      f"over {celltype.reindex(val_ids).nunique()} celltypes")

# %%
