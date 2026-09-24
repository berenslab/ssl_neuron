""" Position augmentations for eyewire2 retina skeletons.

Why these replace the stock augmentations in `ssl_neuron.utils` is written up
in `00_dataset_spec.md` (sections 4 and 6). In short:

* z is a *shared* warped IPL-depth frame and carries most of the celltype
  signal, so it must not be translated, flipped or scaled, and may only be
  jittered by less than the warp's cross-cell alignment error;
* xy arbor size and density are real features, so the in-plane rotation has to
  be an actual rotation -- `ssl_neuron.utils.rotate_graph(axis='z')` is not one.
  It keeps z fixed, but the xy block it leaves behind is the top-left 2x2 of a
  random 3D rotation, which is not orthogonal: it shears and rescales the arbor
  (median area factor 0.50 over 20k draws).

`random_crop_xy` is the one graph-topology augmentation here: it mimics a cell
clipped by the edge of the imaged volume, and returns which nodes survive
rather than new positions.

Everything here is plain numpy, so augmentations can be inspected locally (e.g.
from `02_visualize_data.py`) without the `torch` extra installed.
"""
import numpy as np


def _as_rng(rng):
    """ `None` falls back to numpy's global RNG, which torch's DataLoader
    reseeds per worker -- so dataloader workers do not all draw the same
    augmentations. Pass an explicit `np.random.Generator` for reproducible
    checks. """
    return np.random if rng is None else rng


def random_rotation_xy(positions, rng=None):
    """ Rotate around the z-axis by a uniformly drawn angle (proper SO(2), so
    all distances in the xy-plane are preserved). z is left untouched. """
    rng = _as_rng(rng)
    angle = rng.uniform(0, 2 * np.pi)
    cos, sin = np.cos(angle), np.sin(angle)
    rot = np.array([[cos, -sin], [sin, cos]])

    out = positions.copy()
    out[:, :2] = positions[:, :2] @ rot.T
    return out


def random_mirror_xy(positions, p=0.5, rng=None):
    """ Flip the sign of x with probability `p`. Dendritic arbors have no
    meaningful chirality in the xy-plane, and mirroring combined with
    `random_rotation_xy` covers the full O(2) group. """
    rng = _as_rng(rng)
    if rng.random() >= p:
        return positions.copy()

    out = positions.copy()
    out[:, 0] *= -1
    return out


def random_scale_xy(positions, max_frac, rng=None):
    """ Scale x and y isotropically by a factor in [1 - max_frac, 1 + max_frac].

    Deliberately small: dendritic field size differs systematically between
    celltypes, so this is meant to absorb reconstruction/warping scale noise,
    not to make the model size-invariant. z is never scaled. """
    rng = _as_rng(rng)
    if not max_frac:
        return positions.copy()

    out = positions.copy()
    out[:, :2] *= 1 + rng.uniform(-max_frac, max_frac)
    return out


def random_jitter(positions, sigma, rng=None):
    """ Add per-axis Gaussian noise to every node independently.

    Args:
        sigma: per-axis standard deviations (3,). The z entry should stay at or
            below the cross-cell alignment error of the depth warp.
    """
    rng = _as_rng(rng)
    sigma = np.asarray(sigma, dtype=float)
    if not sigma.any():
        return positions.copy()

    return positions + rng.normal(size=positions.shape) * sigma


def random_translation(positions, sigma, rng=None):
    """ Shift the whole skeleton by one per-axis Gaussian offset.

    Args:
        sigma: per-axis standard deviations (3,). The z entry must be 0 --
            translating along depth moves the cell to a different
            stratification, i.e. to a different celltype.
    """
    rng = _as_rng(rng)
    sigma = np.asarray(sigma, dtype=float)
    if sigma[2] != 0:
        raise ValueError(
            'translate must not have a z component: the z axis is a shared '
            'IPL-depth frame, see 00_dataset_spec.md section 1.2. '
            f'Got {sigma.tolist()}.')
    if not sigma.any():
        return positions.copy()

    return positions + rng.normal(size=3) * sigma


def random_crop_xy(positions, neighbors, size, soma_id=0, margin=0.0, p=1.0,
                   min_nodes=0, max_tries=10, rng=None):
    """ Clip the arbor as if the cell sat near the edge of the imaged volume.

    The volume is modeled as a square of side `size` (µm) in the xy-plane,
    randomly oriented, with the soma placed uniformly inside it but at least
    `margin` µm from every edge. Nodes outside the square are removed, and so
    is everything no longer connected to the soma -- a dendrite that leaves the
    volume and comes back would be a separate fragment in the segmentation, not
    part of this cell. z plays no role.

    This is a graph-topology augmentation (it deletes nodes), so it has to run
    before branch deletion and subsampling, not with the position
    augmentations. Nodes are kept or dropped whole; edges crossing the border
    are not cut at the border, so the clipped arbor ends at its last node
    inside, which on a cached graph with long merged edges can be a few µm
    short of the edge.

    Args:
        positions: node positions (N x 3), xy soma-centered.
        neighbors: dict node id -> set of neighbor ids (not modified).
        size: side length of the square in µm, or a (lo, hi) range to draw it
            from uniformly per call.
        soma_id: node that must survive; the crop is anchored on it.
        margin: minimal distance of the soma to the square's edges, in µm.
            Must be below size / 2. The soma's position is drawn relative to
            the soma itself, so this does not depend on where the cell's other
            nodes are.
        p: probability of cropping at all.
        min_nodes: a crop leaving fewer nodes is redrawn, up to `max_tries`
            times, after which the graph is returned uncropped. Set this to at
            least `n_nodes`, or subsampling has nothing to subsample.
        max_tries: see `min_nodes`.

    Returns:
        The set of kept node ids, or `None` if the graph was left uncropped
        (not drawn, or no draw kept `min_nodes`).
    """
    rng = _as_rng(rng)
    lo, hi = (size, size) if np.isscalar(size) else size
    if not 0 <= margin < lo / 2:
        raise ValueError(f'crop margin must be in [0, size / 2), got margin={margin}, size={size}.')
    if rng.random() >= p:
        return None

    xy = positions[:, :2] - positions[soma_id, :2]
    for _ in range(max_tries):
        side = rng.uniform(lo, hi)
        angle = rng.uniform(0, 2 * np.pi)
        cos, sin = np.cos(angle), np.sin(angle)
        # Node coordinates in the square's frame, with the soma at the origin.
        local = xy @ np.array([[cos, -sin], [sin, cos]])
        # Square's lower corner relative to the soma, per axis.
        corner = -rng.uniform(margin, side - margin, size=2)
        inside = np.all((local >= corner) & (local <= corner + side), axis=1)

        # Soma's connected component within the square.
        kept = {soma_id}
        stack = [soma_id]
        while stack:
            for n in neighbors[stack.pop()]:
                if inside[n] and n not in kept:
                    kept.add(n)
                    stack.append(n)

        if len(kept) >= min_nodes:
            return kept

    return None


def augment_positions(positions,
                      rotate_xy=True,
                      mirror_xy=True,
                      scale_xy=0.0,
                      jitter=(0.0, 0.0, 0.0),
                      translate=(0.0, 0.0, 0.0),
                      rng=None):
    """ Apply the full eyewire2 position augmentation to one view.

    Order is rotate -> mirror -> scale -> jitter -> translate, so that jitter
    and translation magnitudes stay in absolute units (they are not rescaled).

    Args:
        positions: node positions (N x 3), xy soma-centered, z in the shared
            warped depth frame.
        rotate_xy: random rotation around the z-axis.
        mirror_xy: random x-flip.
        scale_xy: max fractional isotropic xy scaling (0 disables).
        jitter: per-axis per-node noise sigma (3,).
        translate: per-axis whole-graph shift sigma (3,); z entry must be 0.

    Returns:
        New (N x 3) array with the input dtype.
    """
    dtype = positions.dtype
    out = positions

    if rotate_xy:
        out = random_rotation_xy(out, rng=rng)
    if mirror_xy:
        out = random_mirror_xy(out, rng=rng)
    out = random_scale_xy(out, scale_xy, rng=rng)
    out = random_jitter(out, jitter, rng=rng)
    out = random_translation(out, translate, rng=rng)

    return out.astype(dtype, copy=False)
