import os
from enum import Enum
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import PIL.Image as PIL
import seaborn as sns
import numpy as np
import torch

from src.utils import get_absolute_coords, read_labels, get_abs_from_rel_batch

# classes defined in data/NAPLabel-LiDAR/names.txt
classes = Enum(
    "classes",
    ["car", "truck", "bus", "motorcycle", "bicycle", "scooter", "person", "rider"],
    start=0,
)
color_class_dict = {
    class_.name: sns.color_palette("hls", len(classes))[i]
    for i, class_ in enumerate(classes)
}


def visualize_predictions(img: PIL.Image | np.ndarray, labels: list, save: bool):
    plt.imshow(img)
    for label in labels:
        width = img.shape[0]
        height = img.shape[1]
        class_, x, y, w, h = get_absolute_coords(label, width, height)
        class_name = classes(class_).name
        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=1,
            edgecolor=color_class_dict[class_name],
            facecolor="none",
        )
        plt.gca().add_patch(rect)
    if save:
        plt.savefig("example.png")
    plt.show()


def visualize_dataset(img: np.ndarray, boxes: torch.Tensor,
                      labels: list, save: bool) -> None:
    """

    :param img: ndarray of shape (H, W, C)
    :param boxes: bounding boxes object with relative xywh coordinates
    :param labels:
    :param save:
    :return:
    """
    plt.imshow(img)
    img_height, img_width = img.shape[:2]
    boxes = get_abs_from_rel_batch(boxes, (img_width, img_height))

    for box, label in zip(boxes, labels):
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2],
            box[3],
            linewidth=1,
            edgecolor=color_class_dict[classes(int(label)).name],
            facecolor="none",
        )
        plt.gca().add_patch(rect)
    if save:
        plt.savefig("example.png")
    plt.show()



# Example usage
img_path = Path("../../data/NAPLab-LiDAR/images/frame_000000.PNG")
label_path = Path("../../data/NAPLab-LiDAR/labels_yolo_v1.1/frame_000000.txt")
if __name__ == "__main__":
    print(os.getcwd())
    visualize_predictions(img_path, read_labels(label_path), save=False)
