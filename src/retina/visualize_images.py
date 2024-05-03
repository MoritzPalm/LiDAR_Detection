import cv2
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