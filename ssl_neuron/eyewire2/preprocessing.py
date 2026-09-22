""" Turning skeliner `.swc` skeletons into the layout `GraphDataset` expects.

The pure functions behind `01_preprocess_data.py`; the notebook itself is the
driver (paths, cell selection, plots, train/val split). Only needs numpy, so it
runs locally on Windows without the `torch` extra.

The coordinate-frame contract this produces is fixed in `00_dataset_spec.md`
section 2: x/y soma-centered in microns, z left untouched in the shared warped
IPL-depth frame, soma as node 0, axons removed, one connected component.
"""
from collections import deque

import numpy as np

from ssl_neuron.data.data_utils import remove_axon

# Skeletons below this many nodes (after axon removal) are broken or partial
# reconstructions rather than small cells -- see 00_dataset_spec.md section 3.
MIN_NODES = 1000


class SkeletonQCError(ValueError):
    """ A skeleton that must not enter the dataset. The message is the reason,
    and `01_preprocess_data.py` tallies these into a drop breakdown. """


def load_swc(path):
    """ Standard 7-column SWC (`id type x y z radius parent`). Node type follows
    the usual convention (1 or -1 = soma, 2 = axon, 3 = dendrite, >=4 = apical
    dendrite); most of these retina skeletons only have soma + dendrite nodes,
    but a few also carry a (partial, unreliable) axon. """
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
    """ One-hot encode SWC type as [soma, axon, dendrite, apical], matching the
    column layout `ssl_neuron.data.data_utils.remove_axon` expects. """
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
    """ Relabel node indices so that node `idx` becomes node 0. `GraphDataset`
    hardcodes the soma as node 0, so this must hold even if the SWC root isn't
    already the first row. """
    if idx == 0:
        return features, neighbors
    order = list(range(len(features)))
    order[0], order[idx] = order[idx], order[0]
    old2new = {old: new for new, old in enumerate(order)}
    new_features = features[order]
    new_neighbors = {old2new[k]: {old2new[v] for v in vs} for k, vs in neighbors.items()}
    return new_features, new_neighbors


def connected_components(neighbors):
    """ Connected components as a list of node-id sets, via BFS.

    `ssl_neuron.data.data_utils.connect_graph` answers the same question with a
    dense N x N adjacency matrix, which these skeletons are far too big for
    (up to ~31k nodes, i.e. a ~7 GB matrix). """
    seen = set()
    components = []
    for start in neighbors:
        if start in seen:
            continue
        component = {start}
        seen.add(start)
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in neighbors[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _as_index(component):
    return np.fromiter(component, dtype=int, count=len(component))


def _closest_pair(positions, source_idx, target_idx, chunk=256):
    """ Closest (squared distance, source, target) node pair between two node
    sets. Chunked over the source nodes to keep the distance matrix bounded
    regardless of component size. """
    best = (np.inf, -1, -1)
    n_chunks = max(1, int(np.ceil(len(source_idx) / chunk)))
    for source_chunk in np.array_split(source_idx, n_chunks):
        dist = ((positions[source_chunk][:, None, :]
                 - positions[target_idx][None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        if dist[i, j] < best[0]:
            best = (dist[i, j], int(source_chunk[i]), int(target_idx[j]))
    return best


def reconnect_components(neighbors, positions, root=0):
    """ Stitch disconnected components onto the component holding `root`, each
    time merging whichever component is spatially closest to what is already
    connected.

    Nearest-first matters: merging in some fixed order (say smallest first)
    can attach a fragment to a far part of the arbor just because the fragment
    that actually sits next to it has not been merged yet, which invents a long
    edge through empty space.

    Same intent as `ssl_neuron.data.data_utils.connect_graph`, but sparse:
    components come from BFS and only candidate pair distances are
    materialized, so this stays usable on 30k-node skeletons. Each component's
    best link is cached and refreshed only against the component just merged,
    since that is the only thing that can improve it.

    Returns:
        neighbors (modified in place) and the number of edges added.
    """
    components = connected_components(neighbors)
    if len(components) <= 1:
        return neighbors, 0

    main = next(c for c in components if root in c)
    pending = [c for c in components if c is not main]
    best = [_closest_pair(positions, _as_index(c), _as_index(main)) for c in pending]

    n_added = 0
    while pending:
        closest = int(np.argmin([b[0] for b in best]))
        _, source, target = best.pop(closest)
        merged = pending.pop(closest)

        neighbors[source].add(target)
        neighbors[target].add(source)
        main |= merged
        n_added += 1

        merged_idx = _as_index(merged)
        for i, component in enumerate(pending):
            candidate = _closest_pair(positions, _as_index(component), merged_idx)
            if candidate[0] < best[i][0]:
                best[i] = candidate

    return neighbors, n_added


def preprocess_cell(swc_path, min_nodes=MIN_NODES):
    """ Full per-cell preprocessing.

    Returns:
        features: node positions (N x 3, float32), xy soma-centered, z untouched.
        neighbors: dict of node id -> set of neighbor ids, soma is node 0.
        soma_xyz: the cell's absolute soma position before centering, kept for
            the later mosaic stage (see 00_dataset_spec.md section 1.4).
        info: per-cell counts for the preprocessing report.

    Raises:
        SkeletonQCError: if the skeleton must not enter the dataset.
    """
    xyz, radii, types, parent_idx = load_swc(swc_path)

    roots = np.where(parent_idx == -1)[0]
    if len(roots) != 1:
        raise SkeletonQCError(f'{len(roots)} root nodes, expected exactly 1')

    features = np.concatenate([xyz, radii[:, None], type_to_onehot(types)], axis=1)  # N x 8
    neighbors = build_neighbors(parent_idx)
    features, neighbors = move_to_front(int(roots[0]), features, neighbors)

    if features[0, 4] != 1:
        raise SkeletonQCError('root node is not typed as soma')

    n_raw = len(features)
    n_axon = int(features[:, 5].sum())

    # Repair connectivity twice. Skeliner's SWC is a tree, so the first pass is
    # usually a no-op -- but axon removal can split the arbor whenever a few
    # nodes along a dendritic path are mislabeled as axon, which is plausible
    # given that these axon labels are unreliable in the first place.
    neighbors, n_stitched_pre = reconnect_components(neighbors, features[:, :3])

    neighbors, features, soma_id = remove_axon(neighbors, features, 0)
    if soma_id != 0:
        raise SkeletonQCError(f'soma ended up at index {soma_id}, expected 0')

    neighbors, n_stitched_post = reconnect_components(neighbors, features[:, :3])

    if len(features) < min_nodes:
        raise SkeletonQCError(f'{len(features)} nodes after axon removal, below {min_nodes}')

    # Center x and y on the soma. z is deliberately left alone: it lives in a
    # depth frame shared across cells, and subtracting the soma's z would
    # re-reference every cell to itself and destroy exactly the signal that
    # separates RGC types (00_dataset_spec.md sections 1.2 and 6.2).
    soma_xyz = features[0, :3].copy()
    features[:, :2] -= soma_xyz[:2]

    info = {
        'n_nodes_raw': n_raw,
        'n_nodes': len(features),
        'n_axon_nodes': n_axon,
        'n_stitched_edges': n_stitched_pre + n_stitched_post,
    }
    return features[:, :3].astype(np.float32), neighbors, soma_xyz, info
