import os.path as osp
import random
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from torch_geometric.data import InMemoryDataset


class ProteinMPNNDataset(InMemoryDataset):
    r"""ProteinMPNN dataset.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        datcut (str, optional): PDB date cutoff.
            (default: :obj:`2030-Jan-01`)
        rescut (float, optional): PDB resolution cutoff.
            (default: :obj:`3.5`)
        homo (float, optional): PDB homo cutoff, to detect homo chains.
            (default: :obj:`0.7`)
        split (str, optional): Which split to use.
            (default: :obj:`train`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #tasks
        * - 130,831
          - ~18.0
          - ~37.3
          - 11
          - 19
    """
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        datcut: str = '2030-Jan-01',
        rescut: float = 3.5,
        homo: float = 0.7,
        split: str = 'train',
    ) -> None:
        assert split in ['train', 'valid', 'test']
        self.datcut = datcut
        self.rescut = rescut
        self.homo = homo
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'pdb_2021aug02/list.csv',
            'pdb_2021aug02/valid_clusters.txt',
            'pdb_2021aug02/test_clusters.txt',
            'pdb_2021aug02/pdb',
        ]

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.split}.pt'

    def download(self) -> None:
        pass

    def _build_clusters(self):
        df = pd.read_csv(self.raw_paths[0])
        df = df[(df['RESOLUTION'] <= self.rescut)
                & (df['DEPOSITION'] <= self.datcut)]
        val_ids = pd.read_csv(self.raw_paths[1], header=None)[0].tolist()
        test_ids = pd.read_csv(self.raw_paths[2], header=None)[0].tolist()
        if self.split == 'valid':
            data = df[df['CLUSTER'].isin(val_ids)]
        elif self.split == 'test':
            data = df[df['CLUSTER'].isin(test_ids)]
        else:
            data = df[~(df['CLUSTER'].isin(val_ids))
                      & ~(df['CLUSTER'].isin(test_ids))]

        return data.groupby('CLUSTER').apply(
            lambda x: list(zip(x['CHAINID'], x['HASH']))).to_dict()

    def _get_pdb(self, data, max_length=10000):
        init_alphabet = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        ]
        extra_alphabet = [str(item) for item in list(np.arange(300))]
        chain_alphabet = init_alphabet + extra_alphabet
        pdb_dict_list = []
        t = {k: v for k, v in data.items()}
        if 'label' in list(t):
            my_dict = {}
            concat_seq = ''
            mask_list = []
            visible_list = []
            if len(list(np.unique(t['idx']))) < 352:
                for idx in list(np.unique(t['idx'])):
                    letter = chain_alphabet[idx]
                    res = np.argwhere(t['idx'] == idx)
                    initial_sequence = "".join(
                        list(np.array(list(t['seq']))[res][
                            0,
                        ]))
                    if initial_sequence[-6:] == "HHHHHH":
                        res = res[:, :-6]
                    if initial_sequence[0:6] == "HHHHHH":
                        res = res[:, 6:]
                    if initial_sequence[-7:-1] == "HHHHHH":
                        res = res[:, :-7]
                    if initial_sequence[-8:-2] == "HHHHHH":
                        res = res[:, :-8]
                    if initial_sequence[-9:-3] == "HHHHHH":
                        res = res[:, :-9]
                    if initial_sequence[-10:-4] == "HHHHHH":
                        res = res[:, :-10]
                    if initial_sequence[1:7] == "HHHHHH":
                        res = res[:, 7:]
                    if initial_sequence[2:8] == "HHHHHH":
                        res = res[:, 8:]
                    if initial_sequence[3:9] == "HHHHHH":
                        res = res[:, 9:]
                    if initial_sequence[4:10] == "HHHHHH":
                        res = res[:, 10:]
                    if res.shape[1] < 4:
                        pass
                    else:
                        my_dict['seq_chain_' + letter] = "".join(
                            list(np.array(list(t['seq']))[res][
                                0,
                            ]))
                        concat_seq += my_dict['seq_chain_' + letter]
                        if idx in t['masked']:
                            mask_list.append(letter)
                        else:
                            visible_list.append(letter)
                        coords_dict_chain = {}
                        all_atoms = np.array(t['xyz'][
                            res,
                        ])[
                            0,
                        ]
                        coords_dict_chain['N_chain_' +
                                          letter] = all_atoms[:,
                                                              0, :].tolist()
                        coords_dict_chain['CA_chain_' +
                                          letter] = all_atoms[:,
                                                              1, :].tolist()
                        coords_dict_chain['C_chain_' +
                                          letter] = all_atoms[:,
                                                              2, :].tolist()
                        coords_dict_chain['O_chain_' +
                                          letter] = all_atoms[:,
                                                              3, :].tolist()
                        my_dict['coords_chain_' + letter] = coords_dict_chain
                my_dict['name'] = t['label']
                my_dict['masked_list'] = mask_list
                my_dict['visible_list'] = visible_list
                my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                my_dict['seq'] = concat_seq
                if len(concat_seq) <= max_length:
                    pdb_dict_list.append(my_dict)
        return pdb_dict_list

    def process(self) -> None:
        clusters = self._build_clusters()

        data_list = []
        for cluster, pdb_list in tqdm(clusters.items()):
            for chain_id, hash_id in pdb_list:
                pdb_id, ch_id = chain_id.split('_')
                prefix = f'{self.raw_paths[3]}/{pdb_id[1:3]}/{pdb_id}'
                # load meta data
                if not osp.isfile(f'{prefix}.pt'):
                    # or return {'seq': np.zeros(5)}
                    continue
                meta = torch.load(f'{prefix}.pt')
                asmb_ids = meta['asmb_ids']
                asmb_chains = meta['asmb_chains']
                ch_ids = np.array(meta['chains'])

                # find candidate assemblies which contain chid chain
                asmb_candidates = {
                    a
                    for a, b in zip(asmb_ids, asmb_chains)
                    if ch_id in b.split(',')
                }

                # if the chains is missing is missing from all the assemblies
                # then return this chain alone
                if len(asmb_candidates) < 1:
                    chain = torch.load(f'{prefix}_{ch_id}.pt')
                    L = len(chain['seq'])
                    data_list.append({
                        'seq': chain['seq'],
                        'xyz': chain['xyz'],
                        'idx': torch.zeros(L).int(),
                        'masked': torch.Tensor([0]).int(),
                        'label': chain_id
                    })
                    continue

                # randomly pick one assembly from candidates
                asmb_i = random.sample(list(asmb_candidates), 1)

                # indices of selected transforms
                idx = np.where(np.array(asmb_ids) == asmb_i)[0]

                # load relevant chains
                chains = {
                    c: torch.load(f'{prefix}_{c}.pt')
                    for i in idx
                    for c in asmb_chains[i] if c in meta['chains']
                }

                # generate assembly
                asmb = {}
                for k in idx:

                    # pick k-th xform
                    xform = meta['asmb_xform%d' % k]
                    u = xform[:, :3, :3]
                    r = xform[:, :3, 3]

                    # select chains which k-th xform should be applied to
                    s1 = set(meta['chains'])
                    s2 = set(asmb_chains[k].split(','))
                    chains_k = s1 & s2

                    # transform selected chains
                    for c in chains_k:
                        try:
                            xyz = chains[c]['xyz']
                            xyz_ru = torch.einsum('bij,raj->brai', u,
                                                  xyz) + r[:, None, None, :]
                            asmb.update({
                                (c, k, i): xyz_i
                                for i, xyz_i in enumerate(xyz_ru)
                            })
                        except KeyError:
                            return {'seq': np.zeros(5)}

                # select chains which share considerable similarity to chid
                seqid = meta['tm'][ch_ids == ch_id][0, :, 1]
                homo = {
                    ch_j
                    for seqid_j, ch_j in zip(seqid, ch_ids)
                    if seqid_j > self.homo
                }
                # stack all chains in the assembly together
                seq, xyz, idx, masked = "", [], [], []
                seq_list = []
                for counter, (k, v) in enumerate(asmb.items()):
                    seq += chains[k[0]]['seq']
                    seq_list.append(chains[k[0]]['seq'])
                    xyz.append(v)
                    idx.append(torch.full((v.shape[0], ), counter))
                    if k[0] in homo:
                        masked.append(counter)

                data_list.append({
                    'seq': seq,
                    'xyz': torch.cat(xyz, dim=0),
                    'idx': torch.cat(idx, dim=0),
                    'masked': torch.Tensor(masked).int(),
                    'label': chain_id
                })

                import pdb
                pdb.set_trace()
        # data_list = []
        # for i, mol in enumerate(tqdm(suppl)):
        #     pass

        #     data = Data(
        #         x=x,
        #         z=z,
        #         pos=pos,
        #         edge_index=edge_index,
        #         smiles=smiles,
        #         edge_attr=edge_attr,
        #         y=y[i].unsqueeze(0),
        #         name=name,
        #         idx=i,
        #     )

        #     if self.pre_filter is not None and not self.pre_filter(data):
        #         continue
        #     if self.pre_transform is not None:
        #         data = self.pre_transform(data)

        #     data_list.append(data)

        # self.save(data_list, self.processed_paths[0])
