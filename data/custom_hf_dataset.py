from typing import List, Tuple
from random import randint

import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from transformers import AutoImageProcessor


def collate_fn(batch):
    return torch.stack([x[0] for x in batch]), torch.tensor(
        [x[1] for x in batch]
    )


class ImageNetHFDataset(Dataset):
    def __init__(self, split):
        self.dataset = load_dataset("imagenet-1k", split=split)

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, idx):
        # load the image and put into torch tensor
        image = self.dataset[idx]["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        processor = AutoImageProcessor.from_pretrained(
            "facebook/deit-base-patch16-224"
        )
        image = processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
        label = self.dataset[idx]["label"]

        return image, label


class ImageNetDataset(Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load the image and put into torch tensor
        image = self.images[idx]
        label = self.labels[idx]

        return image, label


class FakeImageNetDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.num_classes = 1000
        self._len = 1281167
        self.samples: List[Tuple[str, int]] = [
            ("", randint(0, self.num_classes - 1)) for _ in range(self._len)
        ]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        del idx
        image = torch.rand(3, 224, 224)
        label = randint(0, self.num_classes - 1)
        return image, label
