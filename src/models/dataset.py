import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes
import torchvision
import matplotlib.pyplot as plt

from utils import get_relative_coords, read_labels, get_absolute_coords, \
    get_rel_from_abs, get_abs_from_rel_batch, transform, cxcy_to_xy, xy_to_cxcy
from visualization.visualize_img_boxes import visualize_dataset

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
    def __init__(self, img_dir, labels_dir):
        self.img_dir = img_dir
        self.labels_dir = labels_dir

    def __len__(self):
        return len(os.listdir(self.labels_dir)) - 1

    def __getitem__(self, idx: int) -> (
            tuple)[torch.Tensor, tv_tensors.BoundingBoxes, torch.Tensor]:
        """

        :param idx:
        :return: image, bounding boxes in relative xyxy format, labels
        """
        idx = str(idx).zfill(6)  # filling with zeros to match the 6 digit file name
        img_path = os.path.join(self.img_dir, f"frame_{idx}.PNG")
        label_path = os.path.join(self.labels_dir, f"frame_{idx}.txt")
        image = Image.open(img_path)
        image_tensor = tv_tensors.Image(image)
        labels_boxes = read_labels(Path(label_path))
        rel_labels_xywh = []
        labels = []
        for label_box_str in labels_boxes:
            current_label, x, y, w, h = get_relative_coords(label_box_str)
            rel_labels_xywh.append([x, y, w, h])
            labels.append(current_label)
        xyxy_boxes = cxcy_to_xy(torch.tensor(rel_labels_xywh))
        labels = torch.tensor(labels, dtype=torch.long)
        difficulties = torch.zeros(len(labels), dtype=torch.bool)
        image, xyxy_boxes, rel_labels, difficulties = transform(image,
                                                                xyxy_boxes,
                                                                labels, difficulties,
                                                                'TRAIN')
        #image = v2.ToTensor()(image)
        xyxy_boxes = tv_tensors.BoundingBoxes(
            data=xyxy_boxes, format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(image.shape[2], image.shape[1]))
        return image, xyxy_boxes, labels


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

    num_workers = 0

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers,
        collate_fn=collate_fn, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=valid_sampler, num_workers=num_workers,
        collate_fn=collate_fn, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=test_sampler, num_workers=num_workers,
        collate_fn=collate_fn, drop_last=True)
    return train_loader, validation_loader, test_loader


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
    )
    print(f"dataset: {len(dataset)}")
    train_loader, validation_loader, test_loader = make_loaders(dataset,
                                                                batch_size=1,
                                                                validation_split=.2)
    image_batch, boxes_batch, labels_batch = next(iter(train_loader))
    image = image_batch[0]
    boxes = boxes_batch[0]
    boxes = xy_to_cxcy(boxes)
    labels = labels_batch[0]
    visualize_dataset(image.permute(1, 2, 0), boxes, labels, save=False)
    plt.show()
