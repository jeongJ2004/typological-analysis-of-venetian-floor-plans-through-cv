import cv2
import os
import numpy as np

def overlay_images(original_folder, processed_folder, output_folder, alpha=0.1, beta=0.8):
    """
    Overlay wall-extracted images onto original floorplan images, with walls colored red.

    :param original_folder: Folder containing original floorplan images (in color)
    :param processed_folder: Folder containing wall-extracted binary images
    :param output_folder: Folder to save overlay output images
    :param alpha: Weight for the original image
    :param beta: Weight for the wall-extracted image
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(processed_folder):
        if filename.endswith(".jpg"):
            original_path = os.path.join(original_folder, filename)
            processed_path = os.path.join(processed_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".jpg", "_overlay.jpg"))

            # Load both original and wall-extracted images
            original = cv2.imread(original_path, cv2.IMREAD_COLOR)
            processed = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)

            if original is None or processed is None:
                print(f"Failed to load image: {filename}")
                continue

            # Resize processed image if dimensions differ
            if original.shape[:2] != processed.shape[:2]:
                processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

            # Create a 3-channel image for the processed image
            processed_colored = np.zeros((processed.shape[0], processed.shape[1], 3), dtype=np.uint8)

            # Set wall pixels (black in binary, typically 0) to red (BGR: [0, 0, 255])
            wall_mask = processed == 0  # Black pixels (walls) in binary image
            processed_colored[wall_mask] = [0, 0, 255]  # Set walls to red
            processed_colored[~wall_mask] = [255, 255, 255]  # Set non-walls to white

            # Create overlay: weighted sum of original and processed
            overlay = cv2.addWeighted(original, alpha, processed_colored, beta, 0)

            # Save the overlay image
            cv2.imwrite(output_path, overlay)
            print(f"Overlay saved: {output_path}")

if __name__ == '__main__':
    original_folder = "../data/FLOORPLANS_Cut"
    processed_folder = "../results/wall-extracted-imgs-cut"
    output_folder = "../results/overlay-imgs-cut"
    overlay_images(original_folder, processed_folder, output_folder)