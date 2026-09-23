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
# # Render canonical 2D projections for GICLMorph
#
# GICLMorph (`ssl_neuron/giclmorph.py`) aligns the GraphDINO graph embedding with
# a ResNet embedding of deterministic 2D views of the same cell. This notebook
# renders those views from the skeletons `01_preprocess_data.py` wrote, into
# `data/skeletons/<cell_id>/projections.npy` next to `features.npy`, as a
# (V x H x W) uint8 stack.
#
# The default `projection.mode` is `retina`, not the paper's `paper`. Why the
# paper's PCA recipe cannot be applied as-is to RGCs is written up in
# `00_giclmorph_spec.md` section 2; the comparison plot at the end of this
# notebook shows the difference on real cells.
#
# Torch-free, so it runs locally. Run it after `01` (on the cluster: after
# `01` has been re-run there) and before `06_train_giclmorph.py`.
#
# Paths can be overridden with the `GICLMORPH_CONFIG` and `GICLMORPH_DATA`
# environment variables, e.g. for a smoke test on a copy of the data.

# %%
import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ssl_neuron.projection import edges_from_neighbors, render_views, resample_edges

# %% [markdown]
# #### Config

# %%
try:
    THIS_DIR = Path(__file__).resolve().parent
except NameError:
    THIS_DIR = Path.cwd()

CONFIG_PATH = Path(os.environ.get('GICLMORPH_CONFIG', THIS_DIR / 'config_giclmorph.json'))
DATA_DIR = Path(os.environ.get('GICLMORPH_DATA', THIS_DIR / 'data'))
SKELETON_DIR = DATA_DIR / 'skeletons'

with open(CONFIG_PATH) as f:
    config = json.load(f)
PROJ_CFG = dict(config['projection'])
print(f'config: {CONFIG_PATH}')
print(f'data:   {DATA_DIR}')
print(f'projection: {PROJ_CFG}')

cell_ids = [str(c) for split in ('train', 'val')
            for c in np.load(DATA_DIR / f'{split}_ids.npy')]
print(f'{len(cell_ids)} cells in train_ids + val_ids')


def load_cell(cell_id):
    features = np.load(SKELETON_DIR / cell_id / 'features.npy')
    with open(SKELETON_DIR / cell_id / 'neighbors.pkl', 'rb') as f:
        neighbors = pickle.load(f)
    return features[:, :3], neighbors


# %% [markdown]
# #### Stale-data check
#
# Retina mode maps absolute z to image rows, so it is only meaningful if z was
# left in the shared depth frame. Skeletons written before the xy-only
# centering fix have every soma at z = 0 (same check as `02_visualize_data.py`).

# %%
soma_z = np.array([load_cell(c)[0][0, 2] for c in cell_ids[:200]])
if PROJ_CFG['mode'] == 'retina' and np.allclose(soma_z, 0):
    print('WARNING: every soma sits at z = 0 -- these skeletons have soma-centered z. '
          'Re-run 01_preprocess_data.py; retina-mode side views of this data show '
          'depth relative to the soma, not IPL depth.')
else:
    print(f'soma z over {len(soma_z)} cells: median {np.median(soma_z):.1f}, '
          f'range {soma_z.min():.1f} .. {soma_z.max():.1f}')

# %% [markdown]
# #### Dataset-level frame constants (retina mode)
#
# `xy_half_extent` (µm from the soma to the image edge) and `z_range` (µm of
# depth spanned by the image height) must be *global* constants -- a per-cell
# fit would throw away dendritic field size and absolute stratification depth,
# the two things the images are supposed to carry. When they are `null` in the
# config, they are set here from the data, once, and saved alongside the
# images in `projections_meta.json`:
#
# * `xy_half_extent`: 99th percentile across cells of each cell's largest
#   radial distance from the soma, i.e. only the largest 1% of arbors get
#   clipped at the edge;
# * `z_range`: the 0.1 / 99.9 percentiles of pooled node depth, padded by 5%.

# %%
if PROJ_CFG['mode'] == 'retina' and (PROJ_CFG['xy_half_extent'] is None
                                     or PROJ_CFG['z_range'] is None):
    rng = np.random.default_rng(0)
    max_radius, depth_sample = [], []
    for cell_id in tqdm(cell_ids, desc='frame constants'):
        pos, _ = load_cell(cell_id)
        max_radius.append(np.linalg.norm(pos[:, :2] - pos[0, :2], axis=1).max())
        depth_sample.append(rng.choice(pos[:, 2], size=min(2000, len(pos)), replace=False))
    max_radius = np.array(max_radius)
    depth_sample = np.concatenate(depth_sample)

    if PROJ_CFG['xy_half_extent'] is None:
        PROJ_CFG['xy_half_extent'] = float(np.percentile(max_radius, 99))
    if PROJ_CFG['z_range'] is None:
        lo, hi = np.percentile(depth_sample, [0.1, 99.9])
        pad = 0.05 * (hi - lo)
        PROJ_CFG['z_range'] = [float(lo - pad), float(hi + pad)]

    print(f'max radial extent per cell: median {np.median(max_radius):.0f} µm, '
          f'max {max_radius.max():.0f} µm')
    print(f'xy_half_extent = {PROJ_CFG["xy_half_extent"]:.1f} µm '
          f'({2 * PROJ_CFG["xy_half_extent"] / PROJ_CFG["image_size"]:.2f} µm/px)')
    print(f'z_range = {PROJ_CFG["z_range"][0]:.1f} .. {PROJ_CFG["z_range"][1]:.1f} µm '
          f'({np.diff(PROJ_CFG["z_range"])[0] / PROJ_CFG["image_size"]:.2f} µm/px)')

# %% [markdown]
# #### Render every cell

# %%
render_kwargs = dict(mode=PROJ_CFG['mode'], spacing=PROJ_CFG['spacing'],
                     image_size=PROJ_CFG['image_size'], max_count=PROJ_CFG['max_count'],
                     n_views=PROJ_CFG['n_views'])
if PROJ_CFG['mode'] == 'retina':
    render_kwargs.update(en_face=PROJ_CFG['en_face'],
                         xy_half_extent=PROJ_CFG['xy_half_extent'],
                         z_range=tuple(PROJ_CFG['z_range']))

clipped = {}
for cell_id in tqdm(cell_ids, desc='render'):
    pos, neighbors = load_cell(cell_id)
    images, clip = render_views(pos, neighbors, **render_kwargs)
    np.save(SKELETON_DIR / cell_id / 'projections.npy', images)
    clipped[cell_id] = clip

n_views_total = len(next(iter(clipped.values())))
with open(DATA_DIR / 'projections_meta.json', 'w') as f:
    json.dump({'projection': PROJ_CFG, 'n_views_total': n_views_total}, f, indent=2)

clip_max = np.array([c.max() for c in clipped.values()])
print(f'{len(clipped)} cells x {n_views_total} views written')
print(f'fraction of cable outside the image, worst view per cell: median {np.median(clip_max):.4f}, '
      f'{(clip_max > 0.01).sum()} cells above 1%, {(clip_max > 0.1).sum()} above 10%')

# %% [markdown]
# #### Examples
#
# In retina mode the first `n_views` rows are side views (image rows = IPL
# depth, identical across cells) at in-plane angles theta_k from the arbor's
# principal xy direction; the last is the en-face view. Views theta and
# theta + pi are mirror images of each other, as in the paper.

# %%
examples = cell_ids[:4]
fig, axes = plt.subplots(len(examples), n_views_total,
                         figsize=(1.8 * n_views_total, 1.9 * len(examples)), squeeze=False)
for row, cell_id in enumerate(examples):
    images = np.load(SKELETON_DIR / cell_id / 'projections.npy')
    for col, img in enumerate(images):
        axes[row, col].imshow(img, cmap='gray_r', vmin=0, vmax=255)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
    axes[row, 0].set_ylabel(cell_id[-6:], fontsize='x-small')
for col in range(n_views_total):
    title = f'θ={360 * col // PROJ_CFG["n_views"]}°' if col < PROJ_CFG['n_views'] else 'en face'
    axes[0, col].set_title(title, fontsize='small')
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Retina vs. paper mode on the same cells
#
# The paper's recipe (`mode='paper'`): 3D PCA about the centroid, PC1 vertical,
# each cell scaled to fill the image. On a flat RGC arbor PC1 and PC2 both lie
# in the xy-plane, so the paper's first view is en face and depth only shows up
# edge-on, centered on the cell's own centroid and rescaled per cell -- which
# discards both absolute stratification depth and field size.

# %%
fig, axes = plt.subplots(len(examples), 2 * 3, figsize=(11, 1.9 * len(examples)), squeeze=False)
for row, cell_id in enumerate(examples):
    pos, neighbors = load_cell(cell_id)
    paper, _ = render_views(pos, neighbors, mode='paper', spacing=PROJ_CFG['spacing'],
                            image_size=PROJ_CFG['image_size'],
                            max_count=PROJ_CFG['max_count'], n_views=4)
    retina = np.load(SKELETON_DIR / cell_id / 'projections.npy')
    panels = [(paper[0], 'paper θ=0'), (paper[1], 'paper θ=90°'), (paper[2], 'paper θ=180°'),
              (retina[0], f'{PROJ_CFG["mode"]} view 0'),
              (retina[len(retina) // 4], f'{PROJ_CFG["mode"]} view {len(retina) // 4}'),
              (retina[-1], f'{PROJ_CFG["mode"]} last view')]
    for col, (img, title) in enumerate(panels):
        axes[row, col].imshow(img, cmap='gray_r', vmin=0, vmax=255)
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        if row == 0:
            axes[row, col].set_title(title, fontsize='x-small')
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Anisotropy (paper Table 2)
#
# The paper flags cells with anisotropy index AI = 1 - lambda2 / lambda1 below
# 0.1 as PCA failure cases: their principal axis is ill-defined, so the
# "canonical" view is effectively arbitrary. Many RGC arbors are close to
# round, so in retina mode this is the fraction of cells whose in-plane
# reference direction e1 is arbitrary (their side views are still consistent
# in depth, which is what matters most).

# %%
from ssl_neuron.projection import principal_axes  # noqa: E402

ai_xy = []
for cell_id in cell_ids[:500]:
    pos, neighbors = load_cell(cell_id)
    points = resample_edges(pos, edges_from_neighbors(neighbors), PROJ_CFG['spacing'])
    _, _, eigval = principal_axes(points[:, :2] - pos[0, :2])
    ai_xy.append(1 - eigval[1] / eigval[0])
ai_xy = np.array(ai_xy)
print(f'in-plane anisotropy over {len(ai_xy)} cells: median {np.median(ai_xy):.3f}, '
      f'{(ai_xy < 0.1).mean():.1%} below 0.1')

# %%
