import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pybboxes import BoundingBox
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2

from utils import get_relative_coords, read_labels, voc_to_albu, normalize_voc
from visualization.visualize_img_boxes import visualize_dataset

# this prevents errors with too many open files
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
        # only checking labels or boxes should suffice
        if b[0] is None or b[1] is None or b[2] is None:
            # this can lead to huge problems if no boxes are present in the batch
            continue
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
        self.oob_counter = 0

    def __len__(self):
        return len(os.listdir(self.labels_dir)) - 1

    def __getitem__(self, idx: int) -> (
            tuple)[
        torch.Tensor | None, torch.Tensor | None, tv_tensors.BoundingBoxes | None]:
        """

        :param idx:
        :return: image, bounding boxes in absolute xywh format, labels
        """
        idx = str(idx).zfill(6)  # filling with zeros to match the 6 digit file name
        img_path = os.path.join(self.img_dir, f"frame_{idx}.PNG")
        label_path = os.path.join(self.labels_dir, f"frame_{idx}.txt")
        image = tv_tensors.Image(Image.open(img_path))
        labels_boxes = read_labels(Path(label_path))
        voc_boxes = []
        labels = []

        for label_box_str in labels_boxes:
            class_, x, y, w, h = get_relative_coords(label_box_str)
            # converting to voc and checking boundaries
            yolo_box = BoundingBox.from_yolo(x, y, w, h,
                                             (image.shape[2], image.shape[1]),
                                             strict=False)
            if yolo_box.is_oob:
                self.oob_counter += 1
            voc_box = yolo_box.to_voc(return_values=True)
            voc_box = [x - 1 for x in voc_box]  # subtracting 1 to match the 0-based index?? maybe works, probably doesnt
            voc_boxes.append(voc_box)
            labels.append(class_)
        labels = torch.tensor(labels, dtype=torch.long)
        bboxes = tv_tensors.BoundingBoxes(
            torch.tensor(voc_boxes),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(image.shape[1], image.shape[2]),
        )
        bbox_label_dict = {
            "boxes": bboxes,
            "labels": labels,
        }
        if self.transform:
            image, bbox_label_dict = self.transform(image, bbox_label_dict)
        labels = bbox_label_dict.get("labels")
        voc_bboxes = bbox_label_dict.get("boxes")
        if not voc_bboxes.numel():
            return None, None, None
        norm_boxes = normalize_voc(image, voc_bboxes)
        #albu_bboxes = voc_to_albu(voc_bboxes, (image.shape[2], image.shape[1]))
        return image, norm_boxes, labels


def make_loaders(data_path, label_path, train_transform=None,
                 val_transform=None, test_transform=None,
                 batch_size=64, validation_split=0.1, test_split=0.2):
    """
    Splits the dataset into train, validation and test sets
    :param data_path:
    :param label_path:
    :param train_transform:
    :param val_transform:
    :param test_transform:
    :param batch_size:
    :param validation_split:
    :param test_split:
    :return:
    """
    full_dataset = LiDARDataset(data_path, label_path)  # No transform yet

    total_size = len(full_dataset)
    indices = np.arange(total_size)
    np.random.shuffle(indices)

    val_size = int(np.floor(validation_split * total_size))
    test_size = int(np.floor(test_split * total_size))
    train_size = total_size - val_size - test_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_dataset.dataset = LiDARDataset(data_path, label_path, train_transform)
    val_dataset.dataset = LiDARDataset(data_path, label_path, val_transform)
    test_dataset.dataset = LiDARDataset(data_path, label_path, test_transform)

    num_workers = 0

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, collate_fn=collate_fn)
    validation_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=True, collate_fn=collate_fn)

    return train_loader, validation_loader, test_loader


# mean and std from the ImageNet dataset
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float, scale=True),  # this needs to come before Normalize
    #v2.Pad([0, 88, 0, 88], fill=0),  # padding top and bottom to get total size of 300
    v2.Normalize([0, 0, 0], [1, 1, 1]),  # this needs to come after ToDtype
    v2.RandomIoUCrop(),
    v2.SanitizeBoundingBoxes(),
    v2.RandomResizedCrop(size=(300, 300), antialias=True),
    v2.Resize((300, 300)),
    v2.ClampBoundingBoxes(),
    v2.SanitizeBoundingBoxes(),
])

validation_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float, scale=True),
    v2.Normalize([0, 0, 0], [1, 1, 1]),
    v2.RandomResizedCrop(size=(300, 300), antialias=True),
    v2.Resize((300, 300)),
    v2.ClampBoundingBoxes(),
    v2.SanitizeBoundingBoxes(),
    v2.ConvertImageDtype(torch.float32),
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
        transform=train_transforms,
    )
    print(f"dataset: {len(dataset)}")
    (train_loader,
     validation_loader,
     test_loader) = make_loaders(IMAGE_PATH, LABEL_PATH, train_transform=train_transforms,
                                 val_transform=validation_transforms,
                                 test_transform=validation_transforms,
                                 batch_size=1,
                                 validation_split=.2)
    #for i, (image, boxes, labels) in enumerate(train_loader):
    #    print(i)
    #print(f"{dataset.oob_counter=}")
    image_batch, boxes_batch, labels_batch = next(iter(train_loader))

    image = image_batch[0]
    boxes = boxes_batch[0]
    labels = [str(x) for x in labels_batch[0].tolist()]
    visualize_dataset(image.permute(1, 2, 0), boxes, labels, save=False)
    plt.show()
