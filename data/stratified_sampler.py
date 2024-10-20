import random
from typing import Optional
from collections import defaultdict
import math

from torch.utils.data import Sampler
from torch.utils.data.dataset import Dataset
import torch.distributed as dist


class StratifiedSampler(Sampler):
    def __init__(self, dataset, batch_size, num_batches, num_classes):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_classes = num_classes

        self.class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.samples):
            self.class_indices[label].append(idx)

        self.batches = self.create_stratified_batches()

    def create_stratified_batches(self):
        batches = []
        class_indices = list(self.class_indices.values())
        sample_budget = self.batch_size * self.num_batches
        single_sample_size = sample_budget // self.num_classes
        if single_sample_size < 1:
            raise ValueError(
                "Sample budget is too small for the number of classes and batch size."
            )
        sample_count = 0
        for _ in range(self.num_batches):
            if sample_count >= sample_budget:
                break
            batch = []
            for class_idx in class_indices:
                if sample_count >= sample_budget:
                    break
                batch.extend(
                    random.sample(
                        class_idx,
                        single_sample_size,
                    )
                )
                sample_count += single_sample_size
            random.shuffle(batch)
            batches.append(batch)
        return batches

    def __iter__(self):
        flattened_batches = [
            item for sublist in self.batches for item in sublist
        ]
        return iter(flattened_batches)

    def __len__(self):
        return self.num_batches


class DistributedStratifiedSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_batches: int,
        num_classes: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        """Distributed version of stratified sampler

        Args:
            dataset (Dataset): Dataset to use
            batch_size (int): Batch size PER process (total batch_size = batch_size * num_replica)
            num_batches (int): Num of batches, same for all processes.
            num_classes (int): Number of classes in dataset.
            num_replicas (Optional[int], optional): Number of processes (GPUs).
                If None, parsed from torch.distributed utils. Defaults to None.
            rank (Optional[int], optional): Global rank of this process.
                If None, parsed from torch.distributed utils. Defaults to None.
            seed (int, optional): Seed used to set permutation seed. Defaults to 0.

        Raises:
            RuntimeError: If not dist.is_available() and num_replicas or rank is None.
            ValueError: If rank >= num_replicas or rank < 0.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available"
                )
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available"
                )
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_classes = num_classes
        self.seed = seed
        self.num_samples = math.ceil(
            (self.batch_size * self.num_batches) / self.num_replicas
        )
        self.total_size = self.num_replicas * self.num_samples
        self.class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.samples):
            self.class_indices[label].append(idx)
        random.seed(self.seed)
        self.batches = self.create_stratified_batches()

    def create_stratified_batches(self):
        batches = []
        class_indices = list(self.class_indices.values())
        single_sample_size = self.total_size // self.num_classes
        if single_sample_size < 1:
            raise ValueError(
                "Sample budget is too small for the number of classes and batch size."
            )
        sample_count = 0
        for _ in range(self.num_batches):
            if sample_count >= self.total_size:
                break
            batch = []
            for class_idx in class_indices:
                if sample_count >= self.total_size:
                    break
                batch.extend(
                    random.sample(
                        class_idx,
                        single_sample_size,
                    )
                )
                sample_count += single_sample_size
            random.shuffle(batch)
            batches.append(batch)
        return batches

    def __iter__(self):
        flat_batches = []
        for batch in self.batches:
            flat_batches.extend(batch)
        this_proc_flat_batch = flat_batches[
            self.rank : self.total_size : self.num_replicas
        ]
        return iter(this_proc_flat_batch)

    def __len__(self):
        return self.num_samples
