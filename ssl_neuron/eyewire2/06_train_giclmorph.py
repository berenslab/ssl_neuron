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
# # Train GICLMorph on eyewire2 retina skeletons
#
# GraphDINO plus the paper's cross-modal branch: a ResNet-50 embeds one
# canonical 2D projection per cell, and an InfoNCE loss aligns it with the
# student's graph embedding (`ssl_neuron/giclmorph.py`,
# `00_giclmorph_spec.md`). The graph side -- data, augmentations, model,
# schedule -- is identical to `03_train_graphdino.py`, so setting
# `giclmorph.cmid_weight` to 0 in `config_giclmorph.json` gives a GraphDINO run
# with exactly the same budget, which is the control to compare against.
#
# Needs `torch` and a GPU (the ResNet-50 at 224 px roughly doubles the cost of
# a GraphDINO step), so run this on the cluster, after `01` and
# `05_render_projections.py`:
# ```
# uv sync --extra torch
# ```
#
# Paths can be overridden with the `GICLMORPH_CONFIG`, `GICLMORPH_DATA` and
# `GICLMORPH_CKPTS` environment variables.

# %%
import json
import os
from pathlib import Path

import torch

from ssl_neuron.eyewire2.dataset import build_giclmorph_dataloader
from ssl_neuron.giclmorph import create_model
from ssl_neuron.train import GICLTrainer

# %% [markdown]
# #### Config

# %%
try:
    THIS_DIR = Path(__file__).resolve().parent
except NameError:
    THIS_DIR = Path.cwd()

CONFIG_PATH = Path(os.environ.get('GICLMORPH_CONFIG', THIS_DIR / 'config_giclmorph.json'))
with open(CONFIG_PATH) as f:
    config = json.load(f)

config['data']['path'] = str(Path(os.environ.get('GICLMORPH_DATA', THIS_DIR / 'data')))
config['trainer']['ckpt_dir'] = str(Path(os.environ.get(
    'GICLMORPH_CKPTS', THIS_DIR / config['trainer']['ckpt_dir'])))
Path(config['trainer']['ckpt_dir']).mkdir(parents=True, exist_ok=True)

# Keep the config next to the checkpoints, so 07 evaluates with the settings
# the run actually used.
with open(Path(config['trainer']['ckpt_dir']) / 'config.json', 'w') as f:
    json.dump(config, f, indent=4)

if config['trainer'].get('seed') is not None:
    torch.manual_seed(config['trainer']['seed'])

print(f'config: {CONFIG_PATH}')
print(f'data:   {config["data"]["path"]}')
print(f'ckpts:  {config["trainer"]["ckpt_dir"]}')
print(f'giclmorph: {config["giclmorph"]}')

# %% [markdown]
# #### Check the projections match the config
#
# The images are rendered once by `05`; if the projection block was edited
# since, the run would silently train on the old views.

# %%
meta_path = Path(config['data']['path']) / 'projections_meta.json'
if not meta_path.exists():
    raise FileNotFoundError(f'{meta_path} not found -- run 05_render_projections.py first.')
with open(meta_path) as f:
    rendered = json.load(f)['projection']
changed = {k: (rendered.get(k), v) for k, v in config['projection'].items()
           if v is not None and rendered.get(k) != v}
if changed:
    print(f'WARNING: projections were rendered with different settings (rendered, config): {changed}')
print(f'projections: {rendered}')

# %% [markdown]
# #### Model and data

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_model(config)
model.train()
model.to(device)

n_graph = sum(p.numel() for p in model.dino.student_encoder.parameters())
n_image = sum(p.numel() for p in model.image_encoder.parameters())
print(f'device {device}; student graph encoder {n_graph / 1e3:.0f}k params, '
      f'image branch {n_image / 1e6:.1f}M params')

# %%
dataloaders = build_giclmorph_dataloader(config)

train_loader, val_loader = dataloaders
print(f'{len(train_loader.dataset)} train / {len(val_loader.dataset)} val cells, '
      f'{len(train_loader)} iterations per epoch')
print(f'max_iter={config["optimizer"]["max_iter"]} '
      f'(~{config["optimizer"]["max_iter"] // max(len(train_loader), 1)} epochs)')

# %% [markdown]
# #### Run training

# %%
trainer = GICLTrainer(config, model, dataloaders)
trainer.train()

# %%
