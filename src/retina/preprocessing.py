import csv
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt, patches
from torch.utils.data import DataLoader
from torchvision import transforms

from retinanet.dataloader import CSVDataset, Normalizer, Augmenter, Resizer, AspectRatioBasedSampler, collater


def show_bb_of_image():
    with open("data/NAPLab-LiDAR/annotations.csv", "r") as f:
        annotations = f.readlines()

    # Define colors for different classes (you can define your own color scheme)
    class_colors = {"car": (255, 0, 0)}  # For example, blue color for cars

    # Process each annotation
    for annotation in annotations:
        parts = annotation.strip().split(",")
        image_path, x1, y1, x2, y2, class_name = parts
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Read the image
        image = cv2.imread(image_path)

        # Draw bounding box
        color = class_colors.get(class_name, (0, 255, 0))  # Default to green color if class color is not defined
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Show the image with bounding boxes
        cv2.imshow("Image with Bounding Boxes", image)
        cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()


def convert_annotations(dataset_dir, output_file):
    # Load class names from the class_names.txt file
    class_names_file = os.path.join(dataset_dir, "class_names.csv")
    class_id_to_name = {}
    with open(class_names_file, 'r') as classfile:
        for line in classfile:
            class_name, class_id = line.strip().split(',')
            class_id_to_name[int(class_id)] = class_name

    # Open output file for writing
    with open(output_file, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Traverse dataset directory
        for image_file in os.listdir(os.path.join(dataset_dir, "images")):
            if image_file.endswith(".PNG"):  # Assuming images are PNG format
                image_path = os.path.join(dataset_dir, "images", image_file)
                label_path = os.path.join(dataset_dir, "labels", image_file.replace(".PNG", ".txt"))

                # Read label file and parse annotations
                with open(label_path, 'r') as labelfile:
                    for line in labelfile:
                        # Parse relative coordinates and class label
                        class_id, rel_x1, rel_y1, rel_w, rel_h = line.strip().split()
                        class_name = class_id_to_name.get(int(class_id), 'Unknown')
                        rel_x1, rel_y1, rel_w, rel_h = map(float, (rel_x1, rel_y1, rel_w, rel_h))

                        # Get image size
                        image = Image.open(image_path)
                        image_width, image_height = image.size

                        # Convert relative coordinates to absolute pixel coordinates
                        abs_x1 = int(rel_x1 * image_width)
                        abs_y1 = int(rel_y1 * image_height)
                        abs_w = int(rel_w * image_width)
                        abs_h = int(rel_h * image_height)

                        # Calculate absolute bounding box coordinates
                        x1 = int(abs_x1 - (abs_w / 2))
                        y1 = int(abs_y1 - (abs_h / 2))
                        x2 = int(abs_x1 + (abs_w / 2))
                        y2 = int(abs_y1 + (abs_h / 2))

                        # Construct CSV row with forward slashes
                        csvwriter.writerow([image_path.replace("\\", "/"), x1, y1, x2, y2, class_name])


def show_tensor_as_image(tensor):
    # Convert the tensor to a NumPy array and then to PIL Image
    image = tensor.permute(1, 2, 0).cpu().numpy()  # Assuming the tensor is on CPU
    image = (image * 255).astype('uint8')  # Convert back to uint8 range [0, 255]
    image = Image.fromarray(image)
    image.show()


def show_images_with_boxes(data_loader):
    for batch in data_loader:
        images = batch['img']
        annotations = batch['annot']

        for i in range(len(images)):
            image = images[i].permute(1, 2, 0).cpu().numpy()
            annotations_for_image = annotations[i]

            # Plot the image
            plt.imshow(image)
            plt.axis('off')

            # Plot bounding boxes
            for box in annotations_for_image:
                x_min, y_min, x_max, y_max, _ = box.numpy()
                plt.plot([x_min, x_max, x_max, x_min, x_min],
                         [y_min, y_min, y_max, y_max, y_min],
                         color='r', linewidth=2)

            # Show the plot
            plt.show()


def balance_distribution(input_file, output_file, percentage_increase=100):

    # Dictionary to store the counts of each class
    class_counts = {}

    # Read the dataset file and count the occurrences of each class
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            class_name = row[-1]  # Assuming class label is in the last column
            if class_name not in class_counts:
                class_counts[class_name] = 1
            else:
                class_counts[class_name] += 1

    # Find the classes with low representation
    min_count = min(class_counts.values())
    underrepresented_classes = [class_name for class_name, count in class_counts.items() if count < min_count * 1.1]

    # Duplicate entries for underrepresented classes
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            class_name = row[-1]  # Assuming class label is in the last column
            if class_name in underrepresented_classes:
                for _ in range(int(percentage_increase / 100)):
                    writer.writerow(row)
            writer.writerow(row)

    print("Distribution balance completed.")


def main():
    # Path to your dataset directory containing images and labels
    dataset_dir = "data/NAPLab-LiDAR"
    #
    # # Path to save the converted annotations
    output_file = "data/NAPLab-LiDAR/annotations_eval_vis.csv"
    #
    # # Call the function to convert annotations
    convert_annotations(dataset_dir, output_file)

    # Load datafiles
    # csv_classes = "data/NAPLab-LiDAR/class_names.csv"
    # csv_annotations_train = "data/NAPLab-LiDAR/annotations_train.csv"
    # dataset_train = CSVDataset(train_file=csv_annotations_train, class_list=csv_classes,
    #                            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    # sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
    # train_loader = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)
    #
    # show_images_with_boxes(train_loader)
    # balance_distribution("data/NAPLab-LiDAR/annotations_train.csv", "data/NAPLab-LiDAR/annotations_train_balanced.csv", 400)
    # show_tensor_as_image(images[0])


if __name__ == "__main__":
    main()
