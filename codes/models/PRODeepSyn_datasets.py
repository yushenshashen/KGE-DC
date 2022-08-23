import numpy as np
import torch
import random
from torch.utils.data import Dataset

def read_map(map_file):
    d = {}
    with open(map_file, 'r') as f:
        f.readline()
        for line in f:
            k, v = line.rstrip().split('\t')
            d[k] = int(v)
    return d

class FastSynergyDataset(Dataset):

    def __init__(self, drug2id_file, cell2id_file, drug_feat_file, cell_feat_file, synergy_score_file, use_folds,
                 train=True):
        self.drug2id = read_map(drug2id_file)
        self.cell2id = read_map(cell2id_file)
        self.drug_feat = np.load(drug_feat_file)
        self.cell_feat = np.load(cell_feat_file)
        self.samples = []
        self.raw_samples = []
        self.train = train
        valid_drugs = set(self.drug2id.keys())
        valid_cells = set(self.cell2id.keys())
        with open(synergy_score_file, 'r') as f:
            f.readline()
            for line in f:
                drug1, drug2, cellname, score, fold = line.rstrip().split('\t')
                if drug1 in valid_drugs and drug2 in valid_drugs and cellname in valid_cells:
                    if int(fold) in use_folds:
                        #drug1-drug2-cell
                        sample = [
                            torch.from_numpy(self.drug_feat[self.drug2id[drug1]]).float(),
                            torch.from_numpy(self.drug_feat[self.drug2id[drug2]]).float(),
                            torch.from_numpy(self.cell_feat[self.cell2id[cellname]]).float(),
                            torch.FloatTensor([float(score)]),
                        ]
                        self.samples.append(sample)
                        raw_sample = [self.drug2id[drug1], self.drug2id[drug2], self.cell2id[cellname], score]
                        self.raw_samples.append(raw_sample)
                        if train:
                            ###drug2-drug1-cell
                            sample = [
                                torch.from_numpy(self.drug_feat[self.drug2id[drug2]]).float(),
                                torch.from_numpy(self.drug_feat[self.drug2id[drug1]]).float(),
                                torch.from_numpy(self.cell_feat[self.cell2id[cellname]]).float(),
                                torch.FloatTensor([float(score)]),
                            ]
                            self.samples.append(sample)
                            raw_sample = [self.drug2id[drug2], self.drug2id[drug1], self.cell2id[cellname], score]
                            self.raw_samples.append(raw_sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def drug_feat_len(self):
        return self.drug_feat.shape[-1]

    def cell_feat_len(self):
        return self.cell_feat.shape[-1]

    def tensor_samples(self, indices=None):
        if indices is None:
            indices = list(range(len(self)))
        d1 = torch.cat([torch.unsqueeze(self.samples[i][0], 0) for i in indices], dim=0)
        d2 = torch.cat([torch.unsqueeze(self.samples[i][1], 0) for i in indices], dim=0)
        c = torch.cat([torch.unsqueeze(self.samples[i][2], 0) for i in indices], dim=0)
        y = torch.cat([torch.unsqueeze(self.samples[i][3], 0) for i in indices], dim=0)
        return d1, d2, c, y



class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches



