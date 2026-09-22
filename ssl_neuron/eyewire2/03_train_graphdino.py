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
# # Train GraphDINO on eyewire2 retina skeletons
#
# Needs `torch` (a GPU is expected) -- run this on the cluster, not locally:
# ```
# uv sync --extra torch
# ```
# First run `01_preprocess_data.py` there too, pointed at the cluster's own
# (larger) skeleton directory, to populate `./data/skeletons`,
# `./data/train_ids.npy` and `./data/val_ids.npy`.
#
# This uses `RetinaGraphDataset` rather than the stock `GraphDataset`: the
# graph augmentations are identical, but the position augmentations follow the
# retina policy in `00_dataset_spec.md` (xy rotation and mirroring, no
# z-translation, bounded z-jitter).
#
# Two things to watch on the first run:
#
# * **Stale data.** Skeletons preprocessed before the xy-only centering fix
#   have soma-centered z, which silently destroys the depth signal. Run
#   `02_visualize_data.py` first -- it warns if every soma sits at z = 0.
# * **Memory.** `n_nodes` is 512 (was 200) and attention is O(n²), so a step
#   costs ~6.5x what it used to. If this OOMs, drop `batch_size` before
#   dropping `n_nodes`.

# %%
import json
from pathlib import Path

from ssl_neuron.eyewire2.dataset import build_dataloader
from ssl_neuron.graphdino import create_model
from ssl_neuron.train import Trainer

# %% [markdown]
# #### Load config

# %%
try:
    THIS_DIR = Path(__file__).resolve().parent
except NameError:
    THIS_DIR = Path.cwd()

with open(THIS_DIR / "config.json") as f:
    config = json.load(f)

config["data"]["path"] = str(THIS_DIR / "data")
config["trainer"]["ckpt_dir"] = str(THIS_DIR / "ckpts")
Path(config["trainer"]["ckpt_dir"]).mkdir(parents=True, exist_ok=True)

# %% [markdown]
# #### Load model

# %%
model = create_model(config)
model.train()
model.cuda()

# %% [markdown]
# #### Load data

# %%
dataloaders = build_dataloader(config)

train_loader, val_loader = dataloaders
print(f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val cells, "
      f"{len(train_loader)} iterations per epoch")
print(f"max_iter={config['optimizer']['max_iter']} "
      f"(~{config['optimizer']['max_iter'] // max(len(train_loader), 1)} epochs)")

# %% [markdown]
# #### Run training

# %%
trainer = Trainer(config, model, dataloaders)
trainer.train()

# %%
