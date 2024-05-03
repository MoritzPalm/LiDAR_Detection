import numpy as np
import matplotlib.pyplot as plt
import os

from src.models.dataset import make_loaders, transforms, LiDARDataset


class DataVisualizer:
    def __init__(self, dataset, batch_size=64, validation_split=.2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.class_names = [
            "car", "truck", "bus", "motorcycle",
            "bicycle", "scooter", "person", "rider"
        ]

    def make_loaders(self):
        train_loader, validation_loader = make_loaders(
            self.dataset, batch_size=self.batch_size, validation_split=self.validation_split)
        return train_loader, validation_loader

    def plot_class_distribution(self, train_loader, validation_loader):
        train_class_counts = self.calculate_class_counts(train_loader)
        validation_class_counts = self.calculate_class_counts(validation_loader)

        # Combine class counts from both datasets
        class_labels = list(set(list(train_class_counts.keys()) + list(validation_class_counts.keys())))
        train_counts = [train_class_counts.get(label, 0) for label in class_labels]
        validation_counts = [validation_class_counts.get(label, 0) for label in class_labels]

        plt.figure(figsize=(10, 6))
        bar_width = 0.35
        index = np.arange(len(class_labels))

        plt.bar(index, train_counts, bar_width, label='Train Dataset')
        plt.bar(index + bar_width, validation_counts, bar_width, label='Validation Dataset')

        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.title('Class Distribution')
        plt.xticks(index + bar_width / 2, [self.class_names[label] for label in class_labels], rotation=45)
        plt.legend()
        plt.tight_layout()

        # Saving plot
        self.save_plot("class_distribution.png")

    def plot_average_class_counts(self, train_loader, validation_loader):

        train_total_samples = len(train_loader.dataset) * (1 - self.validation_split)
        validation_total_samples = len(validation_loader.dataset) * self.validation_split

        train_avg_counts = self.calculate_avg_class_counts(train_loader, train_total_samples)
        validation_avg_counts = self.calculate_avg_class_counts(validation_loader, validation_total_samples)

        plt.figure(figsize=(10, 6))
        index = np.arange(len(self.class_names))
        bar_width = 0.35

        train_avg_values = [train_avg_counts.get(label, 0) for label in index]
        validation_avg_values = [validation_avg_counts.get(label, 0) for label in index]

        plt.bar(index, train_avg_values, bar_width, label='Train Dataset')
        plt.bar(index + bar_width, validation_avg_values, bar_width, label='Validation Dataset')

        plt.xlabel('Classes')
        plt.ylabel('Average Number of Samples per Image')
        plt.title('Average Class Counts per Image')
        plt.xticks(index + bar_width / 2, self.class_names, rotation=45)
        plt.legend()
        plt.tight_layout()

        # Saving plot
        self.save_plot("average_class_counts.png")

    def plot_class_proportion(self, loader, total_samples, dataset_type):
        # Calculate the average class counts per image
        avg_counts = self.calculate_avg_class_counts(loader, total_samples)

        # Calculate the total counts for each class
        total_counts = {class_: count * total_samples for class_, count in avg_counts.items()}

        # Calculate the total number of classes
        total_classes = sum(total_counts.values())

        # Calculate the proportion of each class
        proportions = {class_: count / total_classes for class_, count in total_counts.items()}

        # Plot the pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(proportions.values(), labels=proportions.keys(), autopct='%1.1f%%', startangle=140)
        plt.title(f'Proportion of Classes in an Average {dataset_type} Picture')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.tight_layout()
        # Save the plot
        self.save_plot(f"{dataset_type}_class_proportion.png")


    def visualize_dataset(self):
        train_loader, validation_loader = self.make_loaders()
        # Plot class distribution for train and validation datasets
        self.plot_class_distribution(train_loader, validation_loader)
        # Plot average class counts for combined dataset
        self.plot_average_class_counts(train_loader, validation_loader)
        # Plot class proportion Trainer
        self.plot_class_proportion(train_loader, len(train_loader.dataset) * (1 - self.validation_split), "Train")
        # Plot class proportion Validation
        self.plot_class_proportion(validation_loader, len(validation_loader.dataset)
                                   * self.validation_split, "Validation")

    def calculate_class_counts(self, data_loader):
        class_counts = {}
        for images, classes, _ in data_loader:
            for class_list in classes:
                for class_ in class_list:
                    if class_ not in class_counts:
                        class_counts[class_] = 1
                    else:
                        class_counts[class_] += 1

        return class_counts

    def save_plot(self, filename):
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.show()

    def calculate_avg_class_counts(self, loader, total_samples):

        # Calculate the average class counts per image
        avg_counts = self.calculate_class_counts(loader)

        avg_counts = {class_: count / total_samples for class_, count in avg_counts.items()}

        return avg_counts


if __name__ == "__main__":
    # Assuming you have a dataset object
    dataset = LiDARDataset(
        "../../data/NAPLab-LiDAR/images",
        "../../data/NAPLab-LiDAR/labels_yolo_v1.1",
        transform=transforms,
    )
    visualizer = DataVisualizer(dataset)
    visualizer.visualize_dataset()
