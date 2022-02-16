#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import numpy as np
import torch.utils.data
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class SymbolicDataset(Dataset):
    """Manually built symbolic dataset."""
    def __init__(self, 
                 n_attributes:int, 
                 n_values:int, 
                 referential=False, 
                 game_size=2, 
                 transform=None,
                 contrastive=False,
    ) -> None:
        self.transform = transform
        self.samples, self.labels = self._build_samples(n_attributes, n_values)
        self.referential = referential
        self.game_size = game_size
        self.contrastive = contrastive
        
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> tuple:
        sample = self.samples[idx]
        generative_label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)
            generative_label = self.transform(generative_label)

        if not self.referential:
            return sample, generative_label

        if self.contrastive:
            return sample, generative_label, sample

        target_sample = sample

        sample_indices = np.random.choice(len(self.samples), self.game_size, replace=False)
        if not idx in sample_indices:
            sample_indices[np.random.randint(self.game_size, size=1)[0]] = idx

        candidates = list(self.samples[sample_indices])

        if self.transform:
            for i in range(len(candidates)):
                candidates[i] = self.transform(candidates[i])

        # build labels for the correct candidate
        assert len(np.where(sample_indices == idx)[0]) == 1
        candidate_label = torch.from_numpy(np.where(sample_indices==idx)[0])

        return target_sample, [generative_label, candidate_label], candidates

    @staticmethod
    def _build_samples(n_a, n_v) -> tuple:
        values = list(np.ndindex(tuple([n_v]*n_a)))
        v_dict = np.eye(n_v)

        sample_matrix = []
        label_matrix = []
        for value in values:
            sample = [v_dict[v] for v in value]
            sample = np.concatenate(sample, axis=0)
            sample_matrix.append(sample)
            label_matrix.append(list(value))

        return np.stack(sample_matrix), np.stack(label_matrix)


def get_symbolic_dataloader(
    n_attributes:int=3,
    n_values:int=6,
    batch_size=64,
    validation_split=.2,
    random_seed=1234,
    shuffle=True,
    referential=False,
    game_size=2,
    contrastive=False,
):
    """
    Key Parameters
        ----------
        n_attributes, n_values:
            The number of attributes and possible values in the manually built dataset, thus there will be 
            $n_{values}^n_{attributes}$ samples in total.
        game_size:
            The number of candidates. If game_size is $n$, then there are $n-1$ distractors for every item.
    """
    dataset = SymbolicDataset(n_attributes, n_values, referential, game_size, torch.FloatTensor, contrastive)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_set = SymbolicDataset(n_attributes, n_values, referential, game_size, torch.FloatTensor, contrastive)
    train_set.samples = dataset.samples[train_indices]
    train_set.labels = dataset.labels[train_indices]
    val_set = SymbolicDataset(n_attributes, n_values, referential, game_size, torch.FloatTensor, contrastive)
    val_set.samples = dataset.samples[val_indices]
    val_set.labels = dataset.labels[val_indices]
    
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=shuffle, batch_size=batch_size)
    validation_loader = torch.utils.data.DataLoader(val_set, shuffle=shuffle, batch_size=batch_size) \
                    if len(val_indices)>0 else None
    
    return train_loader, validation_loader
