""" PCA-guided canonical 2D projections of neuron skeletons, for GICLMorph.

GICLMorph (Hao et al. 2026, "Self-supervised 3D neuronal morphology
representation via graph-image contrastive learning", Expert Systems With
Applications 320, 132059) aligns the GraphDINO graph embedding with a ResNet
embedding of deterministic 2D projections of the same neuron. This module makes
those projections; see `ssl_neuron/giclmorph.py` for the model.

Two modes:

* `paper_views` -- the paper's recipe (section 3.2): resample the skeleton at
  fixed geodesic spacing, center on the centroid, PCA, rotate so that PC1 is
  the image's vertical axis, then rotate about PC1 by theta_k = 2*pi*k/K and
  project onto the plane spanned by PC1 and the rotated PC2. Each cell is
  scaled to fill the image.
* `retina_views` -- the same idea with the biological axis fixed to depth
  instead of learned by PCA. Needed for eyewire2 RGCs; why is written up in
  `ssl_neuron/eyewire2/00_giclmorph_spec.md` section 2.

Everything is plain numpy, so views can be rendered and inspected without the
`torch` extra.
"""
import numpy as np


def edges_from_neighbors(neighbors):
    """ Undirected edge list (E x 2) from a neighbor dict, each edge once. """
    edges = [(i, j) for i, ns in neighbors.items() for j in ns if i < j]
    return np.array(edges, dtype=int).reshape(-1, 2)


def resample_edges(positions, edges, spacing):
    """ Points at (approximately) fixed geodesic spacing along every edge.

    The paper resamples neurites at fixed intervals so that the covariance (and
    the rendered density) reflects arbor length rather than annotation density.
    Each edge of length L contributes ceil(L / spacing) points evenly spread
    along it, starting at its first node, so every point stands for about
    `spacing` of cable. Isolated nodes (no edges) contribute nothing.

    Args:
        positions: node positions (N x 3).
        edges: undirected edges (E x 2).
        spacing: target distance between points, in the units of `positions`.

    Returns:
        (M x 3) float64 array of points.
    """
    start = positions[edges[:, 0]].astype(np.float64)
    delta = positions[edges[:, 1]].astype(np.float64) - start
    length = np.linalg.norm(delta, axis=1)
    counts = np.maximum(1, np.ceil(length / spacing).astype(int))

    edge_idx = np.repeat(np.arange(len(edges)), counts)
    # Position of each point within its edge: 0, 1/n, ..., (n-1)/n.
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    frac = offsets / counts[edge_idx]
    return start[edge_idx] + frac[:, None] * delta[edge_idx]


def _orient_by_skew(axes, coords):
    """ Flip each axis (column) so that the projections onto it have positive
    third moment. PCA eigenvectors have arbitrary sign; without a convention
    the same cell could come out mirrored between runs, and two cells of the
    same shape mirrored against each other. The paper does not say how it
    breaks this tie. """
    proj = coords @ axes
    skew = (proj ** 3).mean(axis=0)
    signs = np.where(skew < 0, -1.0, 1.0)
    return axes * signs


def principal_axes(points):
    """ Principal axes of a point cloud about its centroid.

    Returns:
        centroid (D,), axes (D x D, columns sorted by decreasing variance and
        oriented by `_orient_by_skew`), eigenvalues (D,) in decreasing order.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered / max(len(points) - 1, 1)
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(eigval)[::-1]
    axes = _orient_by_skew(eigvec[:, order], centered)
    return centroid, axes, eigval[order]


def view_angles(n_views):
    """ theta_k = 2*pi*(k-1)/K, k = 1..K (paper eq. 4). """
    return 2 * np.pi * np.arange(n_views) / n_views


def paper_views(points, n_views=6):
    """ The paper's PCA-guided views, in normalized image coordinates.

    PC1 is mapped to the image's vertical axis; view k looks along the
    direction obtained by rotating PC3 about PC1 by theta_k, so its horizontal
    axis is cos(theta_k) PC2 + sin(theta_k) PC3. All views of a cell share one
    isotropic scale, chosen so the cell fills [-1, 1] -- the paper normalizes
    every neuron to [-1, 1], and a shared isotropic factor keeps the aspect
    ratio that the projection is meant to convey.

    Args:
        points: resampled skeleton points (M x 3).
        n_views: K.

    Returns:
        list of K (u, v) coordinate pairs (horizontal, vertical), each (M,).
    """
    centroid, axes, _ = principal_axes(points)
    q = (points - centroid) @ axes  # coordinates in the PCA frame
    scale = np.abs(q).max()
    scale = scale if scale > 0 else 1.0
    q = q / scale

    views = []
    for theta in view_angles(n_views):
        u = np.cos(theta) * q[:, 1] + np.sin(theta) * q[:, 2]
        views.append((u, q[:, 0]))
    return views


def retina_views(points, soma_xy=(0.0, 0.0), n_views=6, en_face=True,
                 xy_half_extent=300.0, z_range=(0.0, 60.0)):
    """ Canonical views for retinal cells in a shared depth frame.

    The biological axis is fixed to depth (z) instead of taken from PCA, and it
    is always the image's vertical axis, mapped through the *global* `z_range`,
    so a pixel row is the same IPL depth in every image. PCA is only used in
    the xy-plane, to pick the in-plane reference direction e1. View k is a side
    view whose horizontal axis is e1 rotated in-plane by theta_k -- i.e. the
    paper's "rotate about the biological axis", with depth as that axis.
    Optionally one en-face view (e1 horizontal, e2 vertical) is appended.

    Horizontal (and en-face) coordinates are soma-centered and divided by the
    global `xy_half_extent`, so dendritic field size survives across cells.

    Args:
        points: resampled skeleton points (M x 3); z in the shared depth frame.
        soma_xy: soma position in xy (the image center).
        n_views: number of side views K.
        en_face: also return an en-face view.
        xy_half_extent: xy distance from the soma that maps to the image edge.
        z_range: (z_min, z_max) that maps to the bottom/top image edge.

    Returns:
        list of (u, v) coordinate pairs in [-1, 1] (points outside fall off
        the image), K side views first, then the en-face view if requested.
    """
    xy = points[:, :2] - np.asarray(soma_xy, dtype=float)
    _, axes, _ = principal_axes(xy)
    e1 = axes[:, 0]

    z_min, z_max = z_range
    v_depth = 2 * (points[:, 2] - z_min) / (z_max - z_min) - 1

    views = []
    for theta in view_angles(n_views):
        cos, sin = np.cos(theta), np.sin(theta)
        direction = np.array([cos * e1[0] - sin * e1[1], sin * e1[0] + cos * e1[1]])
        views.append((xy @ direction / xy_half_extent, v_depth))

    if en_face:
        views.append((xy @ axes[:, 0] / xy_half_extent, xy @ axes[:, 1] / xy_half_extent))
    return views


def render_view(u, v, image_size=224, max_count=64):
    """ Rasterize one view into a grayscale uint8 image.

    Each pixel holds the number of resampled points falling into it, i.e. the
    cable length through that pixel, log-scaled against a *global*
    `max_count` so that brightness is comparable across cells (a per-image
    maximum would make every cell's densest pixel equally bright). Row 0 is
    the top of the image, i.e. v = +1.

    Returns:
        image (image_size x image_size, uint8), and the fraction of points that
        fell outside [-1, 1]^2.
    """
    edges = np.linspace(-1.0, 1.0, image_size + 1)
    counts, _, _ = np.histogram2d(-v, u, bins=[edges, edges])
    inside = counts.sum()
    clipped = 1.0 - inside / max(len(u), 1)
    img = np.clip(np.log1p(counts) / np.log1p(max_count), 0.0, 1.0)
    return (img * 255).round().astype(np.uint8), float(clipped)


def render_views(positions, neighbors, mode='retina', spacing=0.5, image_size=224,
                 max_count=64, n_views=6, **mode_kwargs):
    """ All canonical views of one skeleton, rendered.

    Args:
        positions: node positions (N x 3), soma is node 0.
        neighbors: dict of node id -> neighbor ids.
        mode: 'paper' or 'retina'.
        spacing: geodesic resampling interval, in the units of `positions`.
        image_size, max_count: see `render_view`.
        n_views: K.
        **mode_kwargs: passed on to `paper_views` / `retina_views`.

    Returns:
        images (V x image_size x image_size, uint8) and the per-view clipped
        fraction (V,).
    """
    points = resample_edges(positions, edges_from_neighbors(neighbors), spacing)
    if mode == 'paper':
        views = paper_views(points, n_views=n_views, **mode_kwargs)
    elif mode == 'retina':
        views = retina_views(points, soma_xy=positions[0, :2], n_views=n_views, **mode_kwargs)
    else:
        raise ValueError(f"unknown projection mode {mode!r}, expected 'paper' or 'retina'")

    rendered = [render_view(u, v, image_size=image_size, max_count=max_count) for u, v in views]
    images = np.stack([img for img, _ in rendered])
    clipped = np.array([c for _, c in rendered])
    return images, clipped
