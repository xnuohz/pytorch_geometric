from torch_geometric.datasets import ProteinMPNNDataset


def test_protein_mpnn_dataset():
    dataset = ProteinMPNNDataset(root='./data/ProteinMPNN')
    print(dataset)
    import pdb
    pdb.set_trace()
