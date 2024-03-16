import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import PIL.Image as Image
from enum import Enum
from pathlib import Path

from src.utils import read_labels, get_absolute_coords

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


def visualize_predictions(image_path: Path, labels: list, save: bool):
    img = Image.open(image_path)
    plt.imshow(img)
    for label in labels:
        width = img.size[0]
        height = img.size[1]
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


# Example usage
img_path = Path("../../data/NAPLab-LiDAR/images/frame_000000.PNG")
label_path = Path("../../data/NAPLab-LiDAR/labels_yolo_v1.1/frame_000000.txt")
if __name__ == "__main__":
    print(os.getcwd())
    visualize_predictions(img_path, read_labels(label_path), save=False)
