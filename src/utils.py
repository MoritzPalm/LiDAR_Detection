import random
from enum import Enum
from pathlib import Path
from typing import Tuple

import torch

labels = Enum(
    "classes",
    ["car", "truck", "bus", "motorcycle", "bicycle", "scooter", "person", "rider"],
    start=0,
)
label_map = {k: v + 1 for v, k in enumerate(labels)}
label_map["background"] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ["#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4",
                   "#46f0f0", "#f032e6",
                   "#d2f53c", "#fabebe", "#008080", "#000080", "#aa6e28", "#fffac8",
                   "#800000", "#aaffc3", "#808000",
                   "#ffd8b1", "#e6beff", "#808080", "#FFFFFF"]
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


def read_labels(label_path: Path) -> list[str]:
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
    :param label: string with elements 'class x y w h'
    where x, y are the center of the rectangle
    :return: list with elements [class_, x, y, w, h]
    where x, y are the center of the rectangle
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
    :param label: string with elements 'class x y w h'
    where x, y are the center of the rectangle
    :param img_width: width of the image in pixels
    :param img_height: height of the image in pixels
    :return: list with elements [class_, x, y, w, h]
    where x, y are the top left corner of the rectangle
    """

    class_, rel_x, rel_y, rel_w, rel_h = get_relative_coords(label)
    # scaling all values to the image size
    abs_x = rel_x * img_width
    abs_y = rel_y * img_height
    abs_w = rel_w * img_width
    abs_h = rel_h * img_height

    # getting the top left corner
    x = abs_x - abs_w / 2
    y = abs_y - abs_h / 2
    return class_, x, y, abs_w, abs_h


def get_rel_from_abs(x: float, y: float, w: float, h: float,
                     img_width: int, img_height: int) -> tuple:
    """
    Get the relative coordinates of the bounding box
    :param x: x coordinate of the top left corner
    :param y: y coordinate of the top left corner
    :param w: width of the rectangle
    :param h: height of the rectangle
    :param img_width: width of the image in pixels
    :param img_height: height of the image in pixels
    :return: list with elements [x, y, w, h]
    where x, y are the center of the rectangle
    """
    rel_x = (x + w / 2) / img_width
    rel_y = (y + h / 2) / img_height
    rel_w = w / img_width
    rel_h = h / img_height
    return rel_x, rel_y, rel_w, rel_h


def voc_to_albu(boxes: torch.Tensor, img_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Convert bounding boxes from VOC format (absolute xyxy) to Albumentations format (relative xyxy).
    :param boxes: Tensor of shape (n_boxes, 4) representing the bounding boxes in VOC format.
    :param img_shape: Tuple (img_width, img_height) representing the shape of the image.
    :return: Tensor of shape (n_boxes, 4) representing the bounding boxes in Albumentations format.
    """
    if len(img_shape) != 2:
        raise ValueError("img_shape must be a tuple of length 2.")

    img_width, img_height = img_shape

    if img_width <= 0 or img_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    albu_boxes = boxes.clone().float()  # Create a copy of the input tensor to avoid modifying the original

    albu_boxes[:, 0] /= img_width  # Convert x_min to relative
    albu_boxes[:, 1] /= img_height  # Convert y_min to relative
    albu_boxes[:, 2] /= img_width  # Convert x_max to relative
    albu_boxes[:, 3] /= img_height  # Convert y_max to relative

    if (albu_boxes[:, 2:] < albu_boxes[:, :2]).any():
        raise ValueError("Max values must be greater than min values.")

    return albu_boxes

def get_abs_from_rel_batch(rel_boxes: torch.Tensor, img_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Get the absolute coordinates of a batch of bounding boxes.

    :param rel_boxes: Tensor of shape (batch_size, 4) representing the relative coordinates of boxes.
    :param img_shape: Tuple (img_width, img_height) representing the shape of the image.
    :return: Tensor of shape (batch_size, 4) representing the absolute coordinates of boxes.
    """
    abs_boxes = []
    img_width, img_height = img_shape

    for rel_box in rel_boxes:
        rel_x, rel_y, rel_w, rel_h = rel_box

        if rel_w < 0 or rel_h < 0:
            raise ValueError("Width and height must be non-negative.")

        abs_x = rel_x * img_width - rel_w * img_width / 2
        abs_y = rel_y * img_height - rel_h * img_height / 2
        abs_w = rel_w * img_width
        abs_h = rel_h * img_height

        abs_boxes.append((abs_x, abs_y, abs_w, abs_h))

    return torch.tensor(abs_boxes)


def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

    This is used when we convert FC layers to equivalent
    Convolutional layers, BUT of a smaller size.

    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor;
    None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d),
                                                            step=m[d]).long())

    return tensor


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels,   # noqa: N802
                  true_difficulties, device):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/
    map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors,
    one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors,
    one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors,
    one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors,
    one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors,
    one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors,
    one tensor for each image containing actual objects' difficulty (0 or 1)
    :param device: device on which the tensors are
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(
        true_boxes) == len(
        true_labels) == len(
        true_difficulties)
    # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # Store all (true) objects in a single
    # continuous tensor while keeping track of the image it is from
    true_images = []
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.tensor(true_images, dtype=torch.long, device=device)
    # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single
    # continuous tensor while keeping track of the image it is from
    det_images = []
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.tensor(det_images, dtype=torch.long, device=device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(
        0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1),
                                     dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[
            true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (~true_class_difficulties).sum().item()
        # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)),
                                                dtype=torch.uint8)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0,
                                                descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float)
          # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float)
        # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties,
            # and whether they have been detected before
            object_boxes = true_class_boxes[
                true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[
                true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with
            # objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box,
                                            object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these
            # image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors
            # 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.tensor(range(true_class_boxes.size(0)),
                                        dtype=torch.long, device=device)[
                true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[
                            original_ind] = 1
                        # this object has now been detected/accounted for
                    # Otherwise, it's a false positive
                    # (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different
            # location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall
        # at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives,
                                            dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives,
                                             dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)
        # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects
        # (n_class_detections)

        # Find the mean of the maximum of the precisions
        # corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float, device=device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in
                          enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max)
    to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h)
    to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates,
    a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t.
    the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box,
    and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box,
    and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates,
    a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to
    which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo,
    # completely empirical
    # They are for some sort of numerical conditioning,
    # for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat(
        [(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
         torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model,
    since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes,
    i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which
    the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat(
        [gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
         torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between
    two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to
    each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1),
                             set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1),
                             set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between
    two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to
    each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(
        0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


# Some augmentation functions below have been adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

def expand(image, boxes, filler):
    """
    Perform a zooming out operation
    by placing the image in a larger canvas of filler material.

    Helps to learn to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates,
    a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(
        1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like
    # new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory,
    # so changing one pixel will change all

    # Place the original image at random coordinates
    # in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes


def normalize_voc(image, boxes):
    """
    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.size[1], image.size[0], image.size[1], image.size[0]]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates
    return new_boxes
