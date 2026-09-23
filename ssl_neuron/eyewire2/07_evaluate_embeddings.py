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
# # Evaluate GraphDINO and GICLMorph embeddings on eyewire2 celltypes
#
# Both models are self-supervised; labels enter only here. For every run found
# (GraphDINO from `03`, GICLMorph from `06`, and any extra run named in
# `GICLMORPH_EVAL_RUNS`), the frozen student's CLS embedding of every cell is
# computed and scored the same way:
#
# * **k-NN** (cosine, `knn_k`) and **linear-probe** balanced accuracy, with the
#   labeled cells of `train_ids` as the database / training set and the
#   labeled cells of `val_ids` -- never seen during SSL training -- as queries.
#   k-NN is the headline number (`00_dataset_spec.md` section 8). It is also
#   reported on the consensus-labeled queries alone (`_consensus`), leaving out
#   cells whose celltype was assigned by a classifier;
# * retrieval **precision@k** (`retrieval_k`): the share of a query's `retrieval_k`
#   most similar train cells that share its celltype, macro over classes;
# * the paper's clustering metrics on all labeled cells, using the celltype as
#   the cluster assignment: silhouette (SC, higher is better) and
#   Davies-Bouldin (DBI, lower is better), plus the ARI of a k-means clustering
#   against celltype;
# * the paper's collapse diagnostic (Table 8): the share of embedding variance
#   in the top 10 principal components (E-Top10, lower means less collapse).
#
# A torch-free **depth-profile baseline** (a histogram of node depth plus
# radial field extent, no learning) is scored with the same k-NN, as the floor
# a learned embedding has to beat on RGCs.
#
# Then t-SNE maps per run, and per-class k-NN recall, confusion matrices and
# example retrievals for every run and the baseline.
#
# Needs `torch`; a GPU is not required but helps.

# %%
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

from ssl_neuron.datasets import GraphDataset
from ssl_neuron.giclmorph import create_model as create_giclmorph
from ssl_neuron.graphdino import create_model as create_graphdino
from ssl_neuron.utils import compute_eig_lapl_torch_batch

# %% [markdown]
# #### Config and runs

# %%
try:
    THIS_DIR = Path(__file__).resolve().parent
except NameError:
    THIS_DIR = Path.cwd()

DATA_DIR = Path(os.environ.get('GICLMORPH_DATA', THIS_DIR / 'data'))
with open(Path(os.environ.get('GICLMORPH_CONFIG', THIS_DIR / 'config_giclmorph.json'))) as f:
    EVAL_CFG = json.load(f)['evaluation']

# name -> checkpoint dir. A run's config is read from `<ckpt_dir>/config.json`
# (06 writes it there); runs of 03 have none and fall back to `config.json`.
RUNS = {'GraphDINO': THIS_DIR / 'ckpts', 'GICLMorph': THIS_DIR / 'ckpts_giclmorph'}
if os.environ.get('GICLMORPH_EVAL_RUNS'):
    # e.g. "gamma0=/path/to/ckpts_a;gamma04=/path/to/ckpts_b"
    RUNS = dict(item.split('=', 1) for item in os.environ['GICLMORPH_EVAL_RUNS'].split(';'))
RUNS = {name: Path(d) for name, d in RUNS.items() if list(Path(d).glob('ckpt_*.pt'))}
print(f'data: {DATA_DIR}')
print(f'runs with checkpoints: { {k: str(v) for k, v in RUNS.items()} }')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# #### Labels
#
# From the `cell_meta.csv` sidecar of `01`. Only cells with a valid
# `label_column`, and only classes with at least `min_cells_per_class` labeled
# cells in `train_ids`, are scored.

# %%
LABEL_COLUMN = EVAL_CFG['label_column']
meta = pd.read_csv(DATA_DIR / 'cell_meta.csv', index_col='cell_id', dtype={'cell_id': str})
labeled = meta[LABEL_COLUMN].notna()
if f'valid_{LABEL_COLUMN}' in meta.columns:
    labeled &= meta[f'valid_{LABEL_COLUMN}'].fillna(False).astype(bool)
labels = meta.loc[labeled, LABEL_COLUMN]

split_ids = {split: [str(c) for c in np.load(DATA_DIR / f'{split}_ids.npy')]
             for split in ('train', 'val')}
split_sets = {split: set(ids) for split, ids in split_ids.items()}
train_counts = labels[labels.index.isin(split_ids['train'])].value_counts()
CLASSES = sorted(train_counts[train_counts >= EVAL_CFG['min_cells_per_class']].index)
labels = labels[labels.isin(CLASSES)]
print(f'{len(labels)} labeled cells over {len(CLASSES)} classes '
      f'({(train_counts < EVAL_CFG["min_cells_per_class"]).sum()} classes below '
      f'{EVAL_CFG["min_cells_per_class"]} train cells dropped); '
      f'{labels.index.isin(split_ids["val"]).sum()} of them in val')

# Labels assigned by a classifier rather than by human consensus; scored
# separately (the `_consensus` column). A dataframe without the column counts
# every label as consensus.
if 'celltype_final_decision' in meta.columns:
    consensus_mask = meta['celltype_final_decision'].reindex(labels.index) != 'classifier'
else:
    consensus_mask = pd.Series(True, index=labels.index)
print(f'{int((~consensus_mask).sum())} of the labeled cells carry a classifier-assigned '
      f'label ({int((~consensus_mask[consensus_mask.index.isin(split_ids["val"])]).sum())} in val)')


# %% [markdown]
# #### Scoring

# %%
def rank_neighbors(query, database):
    """ Database indices sorted by cosine similarity to each query, and the
    similarities themselves. """
    q = query / np.linalg.norm(query, axis=1, keepdims=True)
    d = database / np.linalg.norm(database, axis=1, keepdims=True)
    similarity = q @ d.T
    return np.argsort(-similarity, axis=1), similarity


def knn_predict(ranking, database_labels, k):
    votes = database_labels[ranking[:, :k]]
    return np.array([pd.Series(row).mode().iloc[0] for row in votes])


def split_xy(emb_by_cell):
    """ Labeled train cells (the database) and labeled val cells (the queries). """
    tr = [c for c in labels.index if c in split_sets['train'] and c in emb_by_cell]
    va = [c for c in labels.index if c in split_sets['val'] and c in emb_by_cell]
    x_tr = np.stack([emb_by_cell[c] for c in tr])
    x_va = np.stack([emb_by_cell[c] for c in va])
    return x_tr, labels[tr].to_numpy(), x_va, labels[va].to_numpy(), tr, va


def score(emb_by_cell):
    """ All metrics for one {cell_id: vector} embedding. """
    x_tr, y_tr, x_va, y_va, tr, va = split_xy(emb_by_cell)
    k, retrieval_k = EVAL_CFG['knn_k'], EVAL_CFG['retrieval_k']

    out = {'n_train': len(tr), 'n_val': len(va)}
    ranking, _ = rank_neighbors(x_va, x_tr)
    pred = knn_predict(ranking, y_tr, k)
    out[f'{k}nn_bal_acc'] = balanced_accuracy_score(y_va, pred)
    out[f'{k}nn_acc'] = (pred == y_va).mean()
    # The same k-NN, scored only on queries whose label is a human consensus
    # (00_dataset_spec.md section 8). The database keeps every label.
    consensus = consensus_mask.reindex(va).to_numpy()
    out[f'{k}nn_bal_acc_consensus'] = (balanced_accuracy_score(y_va[consensus], pred[consensus])
                                       if consensus.any() else np.nan)
    # Retrieval: of the `retrieval_k` most similar train cells, the share of the
    # query's celltype, macro-averaged over classes.
    hits = (y_tr[ranking[:, :retrieval_k]] == y_va[:, None]).mean(axis=1)
    out[f'P@{retrieval_k}'] = pd.Series(hits).groupby(y_va).mean().mean()

    scaler = StandardScaler().fit(x_tr)
    probe = LogisticRegression(max_iter=5000, class_weight='balanced')
    probe.fit(scaler.transform(x_tr), y_tr)
    out['linear_bal_acc'] = balanced_accuracy_score(y_va, probe.predict(scaler.transform(x_va)))

    x_all = StandardScaler().fit_transform(np.concatenate([x_tr, x_va]))
    y_all = np.concatenate([y_tr, y_va])
    out['SC'] = silhouette_score(x_all, y_all)
    out['DBI'] = davies_bouldin_score(x_all, y_all)
    kmeans = KMeans(n_clusters=len(CLASSES), n_init=10, random_state=0).fit(x_all)
    out['kmeans_ARI'] = adjusted_rand_score(y_all, kmeans.labels_)

    raw = np.stack(list(emb_by_cell.values()))
    eigval = np.linalg.eigvalsh(np.cov(raw - raw.mean(0), rowvar=False))[::-1]
    out['E_top10'] = eigval[:10].sum() / eigval.sum()
    return out


# %% [markdown]
# #### Depth-profile baseline
#
# Per cell: the normalized histogram of node depth over a pooled range (the
# classic stratification profile), plus the log of the 50th/90th percentile
# radial distance from the soma (field size). No learning; if a model does not
# beat this, it has not learned more than stratification and size.

# %%
def load_positions(cell_id):
    return np.load(DATA_DIR / 'skeletons' / cell_id / 'features.npy')[:, :3]


all_ids = split_ids['train'] + split_ids['val']
depth_lo, depth_hi = np.percentile(
    np.concatenate([load_positions(c)[::20, 2] for c in all_ids]), [0.5, 99.5])
bins = np.linspace(depth_lo, depth_hi, 31)

profiles, sizes, mean_depth = [], [], {}
for cell_id in all_ids:
    pos = load_positions(cell_id)
    profiles.append(np.histogram(np.clip(pos[:, 2], depth_lo, depth_hi), bins=bins)[0] / len(pos))
    radius = np.linalg.norm(pos[:, :2] - pos[0, :2], axis=1)
    sizes.append(np.log(np.percentile(radius, [50, 90]) + 1.0))
    mean_depth[cell_id] = pos[:, 2].mean()

# Both blocks scaled across cells to unit spread, so neither dominates the
# cosine similarity by its units alone.
profiles, sizes = np.array(profiles), np.array(sizes)
features = np.concatenate([profiles / profiles.std(),
                           (sizes - sizes.mean(0)) / sizes.std(0)], axis=1)
baseline = dict(zip(all_ids, features))

results = {'depth-profile baseline': score(baseline)}
print(pd.Series(results['depth-profile baseline']).round(3).to_string())


# %% [markdown]
# #### Embed with every run
#
# No augmentation of any kind: no branch dropping, no rotation, jitter or
# translation. The only randomness left is the subsampling to `n_nodes`, so
# each cell's embedding is averaged over `n_eval_views` subsamples (seeded).

# %%
def load_run(ckpt_dir):
    config_path = ckpt_dir / 'config.json'
    if not config_path.exists():
        config_path = THIS_DIR / 'config.json'
    with open(config_path) as f:
        config = json.load(f)
    config['data']['path'] = str(DATA_DIR)

    model = create_giclmorph(config) if 'giclmorph' in config else create_graphdino(config)
    ckpt = max(ckpt_dir.glob('ckpt_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval(), config, ckpt


@torch.no_grad()
def embed(model, config, n_views):
    eval_config = json.loads(json.dumps(config))
    eval_config['data'].update(n_drop_branch=0, jitter_var=0.0, translate_var=0.0,
                               rotation_axis=None)
    torch.manual_seed(0)
    out = {}
    for split in ('train', 'val'):
        dataset = GraphDataset(eval_config, mode=split)
        for i in range(len(dataset)):
            cell = dataset.cells[i]
            views = [dataset._augment(cell) for _ in range(n_views)]
            feat = torch.from_numpy(np.stack([f for f, _ in views])).float().to(device)
            adj = torch.stack([a for _, a in views]).float().to(device)
            lapl = compute_eig_lapl_torch_batch(adj, pos_enc_dim=config['model']['pos_dim'])
            emb, _ = model.student_encoder(feat, adj, lapl)
            out[str(cell['cell_id'])] = emb.mean(0).cpu().numpy()
    return out


embeddings = {}
for name, ckpt_dir in RUNS.items():
    model, config, ckpt = load_run(ckpt_dir)
    print(f'{name}: {ckpt}')
    embeddings[name] = embed(model, config, EVAL_CFG['n_eval_views'])
    results[name] = score(embeddings[name])
    np.save(ckpt_dir / 'embeddings.npy', embeddings[name])

# %% [markdown]
# #### Results
#
# Chance balanced accuracy is 1 / n_classes.

# %%
df_results = pd.DataFrame(results).T
print(f'{len(CLASSES)} classes, chance balanced accuracy {1 / len(CLASSES):.3f}')
print(df_results.round(3).to_string())
df_results.to_csv(DATA_DIR / 'eval_results.csv')

# %% [markdown]
# #### t-SNE per run
#
# Left: celltype (the 10 largest classes in color, the rest grey). Right: mean
# node depth, which needs no labels -- if the embedding has picked up
# stratification, it varies smoothly across the map.

# %%
top = labels.value_counts().index[:10]
for name, emb in embeddings.items():
    ids = list(emb)
    x = np.stack([emb[c] for c in ids])
    xy = TSNE(n_components=2, init='pca', random_state=13,
              perplexity=min(30, max(5, len(x) // 4))).fit_transform(x)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cell_labels = labels.reindex(ids)
    axes[0].scatter(xy[:, 0], xy[:, 1], s=5, c='lightgrey')
    for i, celltype in enumerate(top):
        mask = (cell_labels == celltype).to_numpy()
        axes[0].scatter(xy[mask, 0], xy[mask, 1], s=8, color=plt.cm.tab10(i), label=celltype)
    axes[0].legend(fontsize='x-small', markerscale=2)
    sc = axes[1].scatter(xy[:, 0], xy[:, 1], s=5, c=[mean_depth[c] for c in ids], cmap='coolwarm')
    fig.colorbar(sc, ax=axes[1], label='mean node z')
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(name)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# #### Per-class k-NN recall
#
# The balanced accuracy above is the mean of these bars. Classes are sorted by
# how many labeled train cells they have, so a model that only gets the big
# types right shows up as bars falling off to the right. The baseline is in the
# same plot: the classes where a run beats it are the ones where it learned
# something beyond depth profile and field size.

# %%
by_run = {'depth-profile baseline': baseline, **embeddings}
order = labels[labels.index.isin(split_sets['train'])].value_counts().index
recall = {}
for name, emb in by_run.items():
    x_tr, y_tr, x_va, y_va, _, _ = split_xy(emb)
    pred = knn_predict(rank_neighbors(x_va, x_tr)[0], y_tr, EVAL_CFG['knn_k'])
    recall[name] = pd.Series(pred == y_va).groupby(y_va).mean().reindex(order)
recall = pd.DataFrame(recall)
n_queries = labels[labels.index.isin(split_sets['val'])].value_counts().reindex(order).fillna(0)

fig, ax = plt.subplots(figsize=(max(8, 0.3 * len(order)), 4))
width = 0.8 / len(recall.columns)
for i, name in enumerate(recall.columns):
    ax.bar(np.arange(len(order)) + (i - (len(recall.columns) - 1) / 2) * width,
           recall[name], width=width, label=name)
ax.set_xticks(range(len(order)), [f'{t} ({int(n)})' for t, n in n_queries.items()],
              rotation=90, fontsize='x-small')
ax.set_ylabel(f'{EVAL_CFG["knn_k"]}-NN recall')
ax.set_title('Per-class k-NN recall on val (query count in brackets)')
ax.legend(fontsize='x-small')
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Confusion matrices
#
# Row-normalized, rows = true celltype, same class order as above. Off-diagonal
# mass between types that stratify at the same depth is expected from anything
# that has mostly learned depth; the interesting question is whether a learned
# run resolves pairs the baseline confuses.

# %%
for name, emb in by_run.items():
    x_tr, y_tr, x_va, y_va, _, _ = split_xy(emb)
    pred = knn_predict(rank_neighbors(x_va, x_tr)[0], y_tr, EVAL_CFG['knn_k'])
    confusion = pd.crosstab(pd.Series(y_va, name='true'), pd.Series(pred, name='pred'),
                            normalize='index').reindex(index=order, columns=order).fillna(0)

    fig, ax = plt.subplots(figsize=(0.3 * len(order) + 4, 0.3 * len(order) + 3))
    im = ax.imshow(confusion.to_numpy(), cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize='xx-small')
    ax.set_yticks(range(len(order)), order, fontsize='xx-small')
    ax.set(xlabel='k-NN prediction', ylabel='true celltype', title=name)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# #### Example queries
#
# One val cell per row (black), then its most similar train cells, with their
# cosine similarity; green titles share the query's celltype, red ones do not.
# xz views, because stratification depth is what should be driving the
# similarity -- a red hit at the same depth is a depth-only match.

# %%
N_QUERIES, N_HITS = 4, 5
for name, emb in by_run.items():
    x_tr, y_tr, x_va, y_va, tr, va = split_xy(emb)
    ranking, similarity = rank_neighbors(x_va, x_tr)
    queries = np.random.default_rng(0).choice(len(va), size=min(N_QUERIES, len(va)), replace=False)

    # Shared axes, so field width and depth compare across panels.
    fig, axes = plt.subplots(len(queries), N_HITS + 1, figsize=(2.2 * (N_HITS + 1), 1.6 * len(queries)),
                             sharex=True, sharey=True, squeeze=False)
    for row, q in enumerate(queries):
        pos = load_positions(va[q])
        axes[row, 0].scatter(pos[::5, 0], pos[::5, 2], s=0.3, lw=0, c='k')
        axes[row, 0].set_title(f'query\n{y_va[q]}', fontsize='xx-small')
        for col, hit in enumerate(ranking[q, :N_HITS], start=1):
            pos = load_positions(tr[hit])
            axes[row, col].scatter(pos[::5, 0], pos[::5, 2], s=0.3, lw=0, c='tab:blue')
            axes[row, col].set_title(f'{y_tr[hit]}\n{similarity[q, hit]:.3f}', fontsize='xx-small',
                                     color='tab:green' if y_tr[hit] == y_va[q] else 'tab:red')
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(name)
    plt.tight_layout()
    plt.show()

# %%
