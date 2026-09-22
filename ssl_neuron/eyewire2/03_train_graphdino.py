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

# %%
import json
from pathlib import Path

from ssl_neuron.datasets import build_dataloader
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

# %% [markdown]
# #### Run training

# %%
trainer = Trainer(config, model, dataloaders)
trainer.train()

# %%

# %%
