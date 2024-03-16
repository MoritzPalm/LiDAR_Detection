import os
from pathlib import Path
from PIL import Image
from torchvision import tv_tensors
from torch.utils.data import Dataset

from src.utils import read_labels, get_absolute_coords


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
        img_path = os.path.join(self.img_dir, f"frame_{idx}.npy")
        label_path = os.path.join(self.labels_dir, f"frame_{idx}.npy")
        image = tv_tensors.Image(Image.open(img_path))
        labels = read_labels(Path(label_path))
        width = image.shape[0]
        height = image.shape[1]
        abs_labels = []
        classes = []
        for label in labels:
            class_, x, y, w, h = get_absolute_coords(label, width, height)
            abs_labels.append([x, y, w, h])
        bboxes = tv_tensors.BoundingBoxes(
            abs_labels,
            format=tv_tensors.BoundingBoxFormat.XYWH,
            canvas_size=(width, height),
        )
        target = {"classes": classes, "bboxes": bboxes}
        return image, target
