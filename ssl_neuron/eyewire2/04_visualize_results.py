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
# # Visualize GraphDINO results on eyewire2 retina skeletons
#
# Loads the latest checkpoint written by `03_train_graphdino.py` and checks
# whether the learned embedding is doing what DINO training is supposed to
# do: embeddings of *different augmented views of the same cell* should end
# up close together, while embeddings of *different cells* should be
# further apart.
#
# Needs `torch` (like `03_train_graphdino.py`), so run this on the cluster:
# ```
# uv sync --extra torch
# ```
# A GPU is not required for this notebook (only for training), but it will
# be used automatically if available.

# %%
import json
from pathlib import Path

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from ssl_neuron.datasets import GraphDataset
from ssl_neuron.graphdino import create_model
from ssl_neuron.utils import plot_neuron, plot_tsne

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# #### Load model + latest checkpoint

# %%
ckpt_dir = Path(config["trainer"]["ckpt_dir"])
ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
if not ckpts:
    raise FileNotFoundError(
        f"No checkpoints found in {ckpt_dir}. Run 03_train_graphdino.py first."
    )
ckpt_path = ckpts[-1]
print(f"Loading {ckpt_path}")

model = create_model(config)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
model.to(device)

# %% [markdown]
# #### Load validation data

# %%
val_dataset = GraphDataset(config, mode="val")
n_cells = val_dataset.num_samples
print(f"{n_cells} cells in the validation set")

# %% [markdown]
# #### Plot a few example neurons

# %%
n_examples = min(4, n_cells)
fig, axes = plt.subplots(1, n_examples, figsize=(4 * n_examples, 4), sharex=True, sharey=True)
if n_examples == 1:
    axes = [axes]

for ax, i in zip(axes, range(n_examples)):
    feat, neigh = val_dataset.__getsingleitem__(i)
    plot_neuron(neigh, feat, ax=ax)
    ax.set_title(f"cell {i}\n({len(feat)} nodes)")

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Embed several augmented views per cell
#
# For each validation cell, draw a few independent augmented view pairs
# (random branch dropping + subsampling + rotation/jitter/translation, same
# as during training) and embed each view with the student encoder's CLS
# token (`mlp_head` output, i.e. the first return value of the encoder).

# %%
from ssl_neuron.utils import compute_eig_lapl_torch_batch

n_view_pairs_per_cell = 5

latents = []
cell_labels = []

with torch.no_grad():
    for i in range(n_cells):
        for _ in range(n_view_pairs_per_cell):
            f1, f2, a1, a2 = val_dataset[i]
            feat = torch.from_numpy(np.stack([f1, f2])).float().to(device)
            adj = torch.stack([a1, a2]).float().to(device)
            lapl = compute_eig_lapl_torch_batch(adj, pos_enc_dim=config["model"]["pos_dim"]).float().to(device)

            emb, _ = model.student_encoder(feat, adj, lapl)
            latents.append(emb.cpu().numpy())
            cell_labels.extend([i, i])

latents = np.concatenate(latents, axis=0)
cell_labels = np.array(cell_labels)

# %% [markdown]
# #### t-SNE of the embeddings, colored by cell identity
#
# If training worked, points sharing a color (i.e. views of the same cell)
# should form tight clusters that are separated from other cells' clusters.

# %%
palette = sns.color_palette("husl", n_colors=n_cells)
z = TSNE(n_components=2, perplexity=min(30, max(5, latents.shape[0] // 4))).fit_transform(latents)
plot_tsne(z, cell_labels, targets=[f"cell {i}" for i in range(n_cells)], colors=palette)
plt.title(f"GraphDINO embeddings ({ckpt_path.name})")
plt.show()

# %% [markdown]
# #### Sanity-check metric: within-cell vs. across-cell similarity
#
# Mean cosine similarity between embeddings of the same cell (different
# augmented views) vs. embeddings of different cells. The former should be
# clearly higher than the latter if the model learned to be invariant to
# the augmentations while still telling cells apart.

# %%
normed = latents / np.linalg.norm(latents, axis=1, keepdims=True)
sim = normed @ normed.T
same_cell = cell_labels[:, None] == cell_labels[None, :]
np.fill_diagonal(same_cell, False)

within_sim = sim[same_cell].mean()
across_sim = sim[~same_cell].mean()
print(f"Mean cosine similarity within the same cell:    {within_sim:.3f}")
print(f"Mean cosine similarity across different cells:  {across_sim:.3f}")
