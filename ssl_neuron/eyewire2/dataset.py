""" `GraphDataset` with the eyewire2 retina augmentation policy.

Everything about graph augmentation (random branch deletion, subsampling to a
fixed node count) is inherited unchanged; only the *position* augmentations are
replaced, because the stock ones are wrong for this dataset -- see
`00_dataset_spec.md` sections 4 and 6.
"""
import numpy as np

from ssl_neuron.datasets import GraphDataset, GraphImageDataset, build_dataloader as _build_dataloader
from ssl_neuron.eyewire2.augment import augment_positions


class RetinaGraphDataset(GraphDataset):
    """ Dataset of retinal ganglion cell skeletons.

    Unlike the base class, graphs are soma-centered in x and y *only*: z stays
    in the warped IPL-depth frame that is shared across cells, because
    stratification depth is the main celltype signal and is only meaningful
    relative to the IPL, not relative to each cell's own soma.

    Augmentation parameters come from `config['data']['augment']` and are
    passed straight to `augment_positions`.
    """

    def __init__(self, config, mode='train', inference=False):
        super().__init__(config, mode=mode, inference=inference)

        self.augment_kwargs = dict(config['data'].get('augment', {}))

        # Fail here rather than inside a dataloader worker if the augment block
        # has a typo or asks for a forbidden z-translation.
        augment_positions(np.zeros((2, 3), dtype=np.float32), **self.augment_kwargs)

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
