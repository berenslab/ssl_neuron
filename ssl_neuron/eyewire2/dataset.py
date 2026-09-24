""" `GraphDataset` with the eyewire2 retina augmentation policy.

Everything about graph augmentation (random branch deletion, subsampling to a
fixed node count) is inherited unchanged; only the *position* augmentations are
replaced, because the stock ones are wrong for this dataset -- see
`00_dataset_spec.md` sections 4 and 6.
"""
import numpy as np

from ssl_neuron.datasets import GraphDataset, GraphImageDataset, build_dataloader as _build_dataloader
from ssl_neuron.utils import get_leaf_branch_nodes
from ssl_neuron.eyewire2.augment import augment_positions, random_crop_xy


class RetinaGraphDataset(GraphDataset):
    """ Dataset of retinal ganglion cell skeletons.

    Unlike the base class, graphs are soma-centered in x and y *only*: z stays
    in the warped IPL-depth frame that is shared across cells, because
    stratification depth is the main celltype signal and is only meaningful
    relative to the IPL, not relative to each cell's own soma.

    Augmentation parameters come from `config['data']['augment']` and are
    passed straight to `augment_positions`. An optional `config['data']['crop_xy']`
    block goes to `random_crop_xy`, which clips each view at a simulated
    volume edge before branch deletion and subsampling; `min_nodes` defaults
    to `n_nodes`.
    """

    def __init__(self, config, mode='train', inference=False):
        super().__init__(config, mode=mode, inference=inference)

        self.augment_kwargs = dict(config['data'].get('augment', {}))

        # Fail here rather than inside a dataloader worker if the augment block
        # has a typo or asks for a forbidden z-translation.
        augment_positions(np.zeros((2, 3), dtype=np.float32), **self.augment_kwargs)

        # Volume-edge clipping; absent or null disables it.
        self.crop_kwargs = config['data'].get('crop_xy')
        if self.crop_kwargs:
            self.crop_kwargs = {'min_nodes': self.n_nodes, **self.crop_kwargs}
            random_crop_xy(np.zeros((1, 3)), {0: set()}, **self.crop_kwargs)

    def _augment(self, cell):
        if not self.crop_kwargs:
            return super()._augment(cell)

        soma_id = int(cell['soma_id'])
        kept = random_crop_xy(cell['features'], cell['neighbors'], soma_id=soma_id, **self.crop_kwargs)
        if kept is None:
            return super()._augment(cell)

        # Branch deletion and subsampling then run on the clipped subgraph.
        # Distances to the soma are unchanged on the soma's own component, but
        # the leaf/branch candidates are not: clipping creates new leaves.
        neighbors = {n: cell['neighbors'][n] & kept for n in kept}
        _, adj_matrix, not_deleted = self._reduce_nodes(
            neighbors, [soma_id], cell['distances'], get_leaf_branch_nodes(neighbors))

        features = self._augment_node_position(cell['features'][not_deleted].copy())
        return features, adj_matrix

    def _augment_node_position(self, features):
        features[:, :3] = augment_positions(features[:, :3], **self.augment_kwargs)
        return features


def build_dataloader(config, **kwargs):
    """ Same as `ssl_neuron.datasets.build_dataloader`, with the retina dataset. """
    return _build_dataloader(config, dataset_cls=RetinaGraphDataset, **kwargs)


class RetinaGraphImageDataset(GraphImageDataset, RetinaGraphDataset):
    """ `RetinaGraphDataset` plus one canonical 2D projection per cell, for
    GICLMorph. The graph views are augmented exactly as for GraphDINO; the
    projection is never augmented (the paper draws one of the fixed PCA-guided
    views, and that choice is the only variation). """


def build_giclmorph_dataloader(config, **kwargs):
    """ Same as `build_dataloader`, with projections attached to every item. """
    return _build_dataloader(config, dataset_cls=RetinaGraphImageDataset, **kwargs)
