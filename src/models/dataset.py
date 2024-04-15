import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import tv_tensors
from torchvision.transforms import v2

from utils import get_relative_coords, read_labels

# TODO: check if migration from torchvision to
#  albumentations for bounding box transformations is necessary

# this prevents erros with too many open files
torch.multiprocessing.set_sharing_strategy("file_system")


def collate_fn(batch):
    """
    stacks the images, labels and the bounding boxes
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images,
    lists of varying-size tensors of bounding boxes and labels
    Note: i am unsure if returning lists is the optimal way performance-wise
    """

    images = []
    bboxes = []
    classes = []

    for b in batch:
        images.append(b[0])
        classes.append(b[1])
        bboxes.append(b[2])

    images = torch.stack(images, dim=0)

    return images, classes, bboxes


class LiDARDataset(Dataset):
    def __init__(self, img_dir, labels_dir, transform=None):
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.labels_dir)) - 1

    def __getitem__(self, idx: int) -> (
            tuple)[torch.Tensor, torch.Tensor, tv_tensors.BoundingBoxes]:
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
            classes.append(class_)
        classes = torch.LongTensor(classes)
        bboxes = tv_tensors.BoundingBoxes(
            rel_labels,
            format=tv_tensors.BoundingBoxFormat.XYWH,
            canvas_size=(image.shape[0], image.shape[1]),
        )
        if self.transform:
            image, bboxes = self.transform(image, bboxes)
        return image, classes, bboxes


def make_loaders(dataset, batch_size=64, validation_split=.2) \
        -> tuple[torch.utils.data.DataLoader,
                 torch.utils.data.DataLoader,
                 torch.utils.data.DataLoader]:
    """
    Returns a DataLoader for the given dataset
    :param validation_split: percentage of the dataset to use for validation and testing
    :param dataset:
    :param batch_size:
    :return:
    """
    random_seed = 42
    rng = np.random.default_rng(random_seed)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    split = int(np.floor(validation_split * dataset_size))
    rng.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    split = int(np.floor(validation_split * len(train_indices)))
    rng.shuffle(train_indices)
    train_indices, test_indices = train_indices[split:], train_indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=0, collate_fn=collate_fn)
    validation_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=valid_sampler, num_workers=0, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=test_sampler, num_workers=0, collate_fn=collate_fn)
    return train_loader, validation_loader, test_loader


transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((300, 300)),
    v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

if __name__ == "__main__":
    DATA_PATH = "../../data/NAPLab-LiDAR"
    IMAGE_PATH = "../../data/NAPLab-LiDAR/images"
    LABEL_PATH = "../../data/NAPLab-LiDAR/labels_yolo_v1.1"
    with open(f"{DATA_PATH}/train.txt", "r") as f:
        train_files = f.read().split("\n")
        train_files.remove("")

    dataset = LiDARDataset(
        "../../data/NAPLab-LiDAR/images",
        "../../data/NAPLab-LiDAR/labels_yolo_v1.1",
        transform=transforms,
    )
    print(f"dataset: {len(dataset)}")
    train_loader, validation_loader, test_loader = make_loaders(dataset,
                                                                batch_size=1,
                                                                validation_split=.2)
    imgs_list = []
    classes_list = []
    bboxes_list = []
    for i, (img, classes, bboxes) in enumerate(train_loader):
        imgs_list.extend(img)
        classes_list.extend(classes)
        bboxes_list.extend(bboxes)
    print(f"img: {len(imgs_list)}, classes: {len(classes_list)}, "
          f"bboxes: {len(bboxes_list)}")
