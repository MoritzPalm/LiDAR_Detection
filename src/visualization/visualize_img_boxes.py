import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import PIL.Image as Image
from enum import Enum
from pathlib import Path

# classes defined in data/NAPLabel-LiDAR/names.txt
classes = Enum('classes', ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider'], start=0)
color_class_dict = {class_.name: sns.color_palette('hls', len(classes))[i] for i, class_ in enumerate(classes)}


def read_label(image_path: Path):
    label_path = image_path.parent.parent / 'labels_yolo_v1.1' / f'{image_path.stem}.txt'
    with open(label_path, 'r') as f:
        labels = f.read().split('\n')
        if '' in labels:
            labels.remove('')
    return labels


def get_absolute_coords(label: str, img: Image.Image) -> tuple:
    """
    Get the absolute coordinates of the bounding box
    :param label: string with elements 'class x y w h' where x, y are the center of the rectangle
    :param img: PIL.Image object
    :return: list with elements [class_, x, y, w, h] where x, y are the top left corner of the rectangle
    """
    label = label.split(' ')
    class_ = int(label[0])
    # scaling all values to the image size
    x = float(label[1]) * img.size[0]  # this is the center of the rectangle
    y = float(label[2]) * img.size[1]  # this is the center of the rectangle
    w = float(label[3]) * img.size[0]
    h = float(label[4]) * img.size[1]

    # getting the top left corner
    x = x - w / 2
    y = y - h / 2
    return class_, x, y, w, h


def visualize_predictions(image_path: Path, labels: list, save: bool):
    img = Image.open(image_path)
    imgplot = plt.imshow(img)
    for label in labels:
        class_, x, y, w, h = get_absolute_coords(label, img)
        class_name = classes(class_).name
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color_class_dict[class_name], facecolor='none')
        plt.gca().add_patch(rect)
    if save:
        plt.savefig('example.png')
    plt.show()


# Example usage
img_path = Path('../../data/NAPLab-LiDAR/images/frame_000000.PNG')
if __name__ == '__main__':
    print(os.getcwd())
    visualize_predictions(img_path, read_label(img_path), save=False)
