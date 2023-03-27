from pathlib import Path

import torch

from .sequence import Sequence

class DatasetProvider:
    def __init__(self, dataset_path: Path, delta_t_ms: int=50, num_bins=5):

        train_path = dataset_path / 'train'
        
        assert dataset_path.is_dir(), str(dataset_path)
        assert train_path.is_dir(), str(train_path)

        train_sequences = list()
        for child in train_path.iterdir():

            train_sequences.append(Sequence(child, 'train', delta_t_ms, num_bins))

        self.train_dataset = torch.utils.data.ConcatDataset(train_sequences)

        val_path = dataset_path / 'val'
        
        assert val_path.is_dir(), str(val_path)

        val_sequences = list()
        for child in val_path.iterdir():

            val_sequences.append(Sequence(child, 'val', delta_t_ms, num_bins))

        
        self.val_dataset = torch.utils.data.ConcatDataset(val_sequences)

        test_path = dataset_path / 'test'
        
        assert test_path.is_dir(), str(test_path)

        test_sequences = list()
        for child in test_path.iterdir():

            test_sequences.append(Sequence(child, 'test', delta_t_ms, num_bins))

        
        self.test_dataset = torch.utils.data.ConcatDataset(test_sequences)

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        # Implement this according to your needs.
        return self.val_dataset
        # raise NotImplementedError

    def get_test_dataset(self):
        # Implement this according to your needs.
        # raise NotImplementedError
        return self.test_dataset
