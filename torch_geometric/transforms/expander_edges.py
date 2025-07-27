import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('add_expander_edges')
class AddExpanderEdges(BaseTransform):
    r"""Generates a random 2d-regular graph with n nodes
    using permutations algorithm.
    Returns the list of edges. This list is symmetric; i.e., if
    (x, y) is an edge so is (y,x).

    Args:
        degree: Desired degree.

    Returns:
        data: Data object with added expander edges.
    """
    def __init__(self, degree: int = 3):
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
        expand_edge_index = torch.stack([senders, receivers], dim=0)

        combined_edges = torch.cat([data.edge_index, expand_edge_index], dim=1)
        edges_set = set()
        for i in range(combined_edges.shape[1]):
            node1, node2 = combined_edges[0,
                                          i].item(), combined_edges[1,
                                                                    i].item()
            edges_set.add((node1, node2))
        data.edge_index = torch.tensor(list(edges_set)).t()

        return data
