import torch

from torch_geometric.data import Data
from torch_geometric.transforms import AddExpanderEdges


def test_add_expander_edges():
    transform = AddExpanderEdges()

    # Directed.
    edge_index = torch.tensor([
        [0, 1, 2, 2, 3],
        [1, 2, 0, 3, 0],
    ])
    data = Data(edge_index=edge_index, num_nodes=4)
    data = transform(data)
    assert data.edge_index.size() == (2, 6)
