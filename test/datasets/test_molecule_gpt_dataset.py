from torch_geometric.datasets import MoleculeGPTDataset


def test_fake_dataset():
    dataset = MoleculeGPTDataset(root='./data/MoleculeGPT')

    assert str(dataset) == f'MoleculeGPTDataset({len(dataset)})'
    assert dataset.num_edge_features == 4
    assert dataset.num_node_features == 5