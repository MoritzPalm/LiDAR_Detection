import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import tv_tensors, transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from src.utils import read_labels, get_relative_coords

# TODO: include data augmentation and other transformations
# TODO: add image reshape and normalization


class LiDARDataset(Dataset):
    def __init__(self, img_dir, labels_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx: int):
        idx = str(idx).zfill(6)  # filling with zeros to match the 6 digit file name
        img_path = os.path.join(self.img_dir, f"frame_{idx}.PNG")
        label_path = os.path.join(self.labels_dir, f"frame_{idx}.txt")
        image = tv_tensors.Image(Image.open(img_path))
        labels = read_labels(Path(label_path))
        rel_labels = []
        classes = []
        for label in labels:
            class_, x, y, w, h = get_relative_coords(label)
            rel_labels.append([x, y, w, h])
        bboxes = tv_tensors.BoundingBoxes(
            rel_labels,
            format=tv_tensors.BoundingBoxFormat.XYWH,
            canvas_size=(image.shape[0], image.shape[1]),
        )
        target = {"classes": classes, "bboxes": bboxes}
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target


transform = transforms.Compose(
    [
        #transforms.Resize((256, 256)),
        #transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

target_transform = transforms.Compose(
    [
        #transforms.ToTensor(),
    ]
)


def make_loaders(dataset, batch_size=64, validation_split=.2) \
        -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Returns a DataLoader for the given dataset
    :param validation_split:
    :param dataset:
    :param batch_size:
    :return:
    """
    random_seed = 42
    np.random.seed(random_seed)
    validation_split = .2

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, shuffle=False)
    return train_loader, validation_loader
