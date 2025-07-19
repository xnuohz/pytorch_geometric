import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('random_regular_expander_edges')
class RandomRegularExpanderEdges(BaseTransform):
    r"""Generates a random 2d-regular graph with n nodes
    using permutations algorithm.
    Returns the list of edges. This list is symmetric; i.e., if
    (x, y) is an edge so is (y,x).

    Args:
            degree: Desired degree.

    Returns:
            data: Data.
    """
    def __init__(self, degree: int = 4):
        self.degree = degree

    def forward(self, data: Data) -> Data:
        import numpy as np

        assert data.num_nodes is not None
        num_nodes = data.num_nodes
        rng = np.random.default_rng()

        senders = [*range(0, num_nodes)] * min(self.degree, num_nodes - 1)
        receivers = rng.permutation(senders).tolist()

        senders, receivers = [*senders, *receivers], [*receivers, *senders]

        # eliminate self loops.
        non_loops = [
            *filter(lambda i: senders[i] != receivers[i], range(
                0, len(senders)))
        ]

        senders = np.array(senders)[non_loops]
        receivers = np.array(receivers)[non_loops]
        senders = torch.tensor(senders, dtype=torch.long)
        receivers = torch.tensor(receivers, dtype=torch.long)
        data.expand_edge_index = torch.stack([senders, receivers], dim=0)
        import pdb
        pdb.set_trace()
        return data
