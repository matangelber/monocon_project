import cv2
import numpy as np
import os

def create_stereo_photo_viewer(left_folder_path, right_folder_path):
    """
    Creates a stereo photo viewer that displays two images of a stereo pair side by side.
     Clicking the right or left arrow keys navigates through the images.

    Args:
        left_folder_path (str): The path to the folder containing the left stereo images.
        right_folder_path (str): The path to the folder containing the right stereo images.
    """

    # Get a list of image filenames in both folders
    left_image_files = [f for f in os.listdir(left_folder_path) if f.endswith('.png')]  # Adjust for your image format
    right_image_files = [f for f in os.listdir(right_folder_path) if f.endswith('.png')]

    # Ensure both folders have the same number of images
    if len(left_image_files) != len(right_image_files):
        print("Error: The number of images in the left and right folders does not match.")
        return

    # Create a window for the stereo photo viewer
    cv2.namedWindow('Stereo Photo Viewer', cv2.WINDOW_NORMAL)

    # Initialize current image index
    current_index = 0

    while True:
        # Load the current image pair
        image_path_left = os.path.join(left_folder_path, left_image_files[current_index])
        image_path_right = os.path.join(right_folder_path, right_image_files[current_index])

        img_left = cv2.imread(image_path_left)
        img_right = cv2.imread(image_path_right)

        if img_left is None or img_right is None:
            print(f"Error loading image: {image_path_left}")
            break

        # # Resize images if necessary (adjust dimensions as needed)
        # img_left = cv2.resize(img_left, (640, 480))
        # img_right = cv2.resize(img_right, (640, 480))

        # Combine the two images vertically
        combined_image = np.vstack([img_left, img_right])

        # Display the combined image
        cv2.imshow('Stereo Photo Viewer', combined_image)

        # Handle key presses
        key = cv2.waitKey(1)
        if key == 27:  # Esc key to exit
            break
        elif key == ord('d'):  # Right arrow key to go forward
            current_index = (current_index + 1) % len(left_image_files)
        elif key == ord('a'):  # Left arrow key to go backward
            current_index = (current_index - 1) % len(left_image_files)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    left_folder_path = "/home/matan/Projects/MonoCon/mmdetection3d-0.14.0/data/kitti/training/image_2"  # Replace with the actual left folder path
    right_folder_path = "/home/matan/Projects/MonoCon/mmdetection3d-0.14.0/data/kitti/training/image_3"  # Replace with the actual right folder path
    create_stereo_photo_viewer(left_folder_path, right_folder_path)