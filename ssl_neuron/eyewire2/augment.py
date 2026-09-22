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
