from torch.utils.data import Sampler
from src.dataset import MultiDataset
import random

class SamplerWithUnlabeled(Sampler[int]):
    def __init__(self,
                 dataset: MultiDataset,
                 unlabeled_samples_share: float,
    ):
                 
        super().__init__(dataset)
        self.data_source = dataset
        self.unlabeled_samples_share = unlabeled_samples_share

        self.total_samples = self.data_source.size
        self.labeled_samples = self.data_source.labeled_size
        self.unlabeled_samples = self.total_samples - self.labeled_samples

        self.num_unlabeled_samples = min(int(self.unlabeled_samples_share * self.labeled_samples), self.unlabeled_samples)
        self.num_labeled_samples = self.labeled_samples

        self.num_samples = self.num_labeled_samples + self.num_unlabeled_samples

        print(f"Using {self.num_labeled_samples} labeled samples and {self.num_unlabeled_samples} unlabeled samples")

    def __iter__(self):
        labeled_indices = list(range(self.labeled_samples))
        unlabeled_indices = list(range(self.labeled_samples, self.total_samples))

        random.shuffle(labeled_indices)
        random.shuffle(unlabeled_indices)

        all_indices = labeled_indices + unlabeled_indices[:self.num_unlabeled_samples]
        random.shuffle(all_indices)
        yield from all_indices

    def __len__(self):
        return self.num_samples
