from pathlib import Path


def read_labels(label_path: Path) -> list:
    """
    Read the labels from the .txt file for one image
    :param label_path: Path object to the image
    :return: list of strings with the labels
    """
    with open(label_path, "r") as f:
        labels = f.read().split("\n")
        if "" in labels:
            labels.remove("")
    return labels


def get_relative_coords(label: str) -> tuple:
    """
    Get the relative coordinates of the bounding box
    :param label: string with elements 'class x y w h' where x, y are the center of the rectangle
    :return: list with elements [class_, x, y, w, h] where x, y are the center of the rectangle
    """
    label = label.split(" ")
    class_ = int(label[0])
    x = float(label[1])
    y = float(label[2])
    w = float(label[3])
    h = float(label[4])
    return class_, x, y, w, h


def get_absolute_coords(label: str, img_width: int, img_height: int) -> tuple:
    """
    Get the absolute coordinates of the bounding box
    :param label: string with elements 'class x y w h' where x, y are the center of the rectangle
    :param img_width: width of the image in pixels
    :param img_height: height of the image in pixels
    :return: list with elements [class_, x, y, w, h] where x, y are the top left corner of the rectangle
    """

    class_, x, y, w, h = get_relative_coords(label)
    # scaling all values to the image size
    x = x * img_width  # this is the center of the rectangle
    y = y * img_height  # this is the center of the rectangle
    w = w * img_width
    h = h * img_height

    # getting the top left corner
    x = x - w / 2
    y = y - h / 2
    return class_, x, y, w, h
