import cv2
import numpy as np
import os

def load_image(img_path):
    """
    Load an image in grayscale or color mode.

    :param img_path: Path to the image file
    :return: Image as a NumPy array
    :raises ValueError: If the image can't be loaded
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f'Could not load image: {img_path}')
    return img


def remove_text_and_numbers(binary_img):
    """
    Remove text and numbers from a binary floor plan image,
    keeping only the walls (thick black lines).

    :param binary_img: Binary image (walls are black, background is white)
    :return: Cleaned image with only walls
    """
    # Invert image for morphological operations (walls become white)
    inverted = cv2.bitwise_not(binary_img)

    # Define kernels for morphological operations
    # Horizontal kernel to remove horizontal text
    h_kernel = np.ones((1, 4), np.uint8)
    # Vertical kernel to remove vertical text
    v_kernel = np.ones((4, 1), np.uint8)
    # Small circular kernel to remove dots and small elements
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # The kernel defines a neighborhood around each pixel.
                                                                              # For an ellipse, it’s roughly circular, with 1s in the middle and 0s at the edges
                                                                              # (approximated within a 3x3 grid).

    # First, perform opening to remove small elements (text, dots, etc.)
    opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, small_kernel, iterations=1)

    # Remove thin horizontal and vertical lines (often text or small details)
    h_opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_opened = cv2.morphologyEx(h_opened, cv2.MORPH_OPEN, v_kernel, iterations=1)

    # Apply area filtering to remove small connected components (likely text)
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(v_opened, connectivity=8) #connectivity=8 means it checks the 8 sides around itself (left, right, up, down, diagonals)

    # Create output image
    filtered = np.zeros_like(v_opened)

    # Set minimum area threshold (adjust as needed based on your images)
    min_area = 30  # Areas smaller than this will be removed

    '''
    Loop : Checks each component (skips background, i=0).
    Area check : If area > 30, it’s a wall (not text).
    component_mask :
        (labels == i): Boolean array (True where pixel belongs to component i).
        .astype(np.uint8) * 255: Converts to 0/255 (black/white) binary mask.
        Result : A mask where only this component is white.
    cv2.bitwise_or(filtered, component_mask) :
        OR operation : Adds the white component to filtered. If a pixel is white in either filtered or component_mask, it stays white.
    '''

    # Process each connected component
    for i in range(1, num_labels):  # Start from 1 to skip background
        area = stats[i, cv2.CC_STAT_AREA] # extracts the size of each white region. If it’s less than min_area (30 pixels), it’s considered noise/text and discarded.

        # Keep only components with area larger than threshold
        if area > min_area:
            component_mask = (labels == i).astype(np.uint8) * 255
            filtered = cv2.bitwise_or(filtered, component_mask)

    # Invert back to original format (walls are black)
    result = cv2.bitwise_not(filtered)

    return result

def save_result(img, output_path):
    """
    Save the processed image.

    :param img: Processed image
    :param output_path: Path to save the image
    """
    cv2.imwrite(output_path, img)
    print(f"Saved image to {output_path}")


def process_wall_extraction():
    thresholded_folder = "../results/thresholded-imgs-cut"
    wall_extracted_folder = "../results/wall-extracted-imgs-cut"
    os.makedirs(wall_extracted_folder, exist_ok=True)

    for filename in os.listdir(thresholded_folder):
        if filename.endswith(".jpg"):
            input_path = os.path.join(thresholded_folder, filename)
            binary_img = load_image(input_path)

            # Convert to grayscale if it's not already
            if len(binary_img.shape) > 2:
                binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)

            # Ensure binary image (0 or 255)
            _, binary_img = cv2.threshold(binary_img, 128, 255, cv2.THRESH_BINARY)

            # Remove text and numbers
            walls_only = remove_text_and_numbers(binary_img)

            # Save the result
            output_path = os.path.join(wall_extracted_folder, filename)
            save_result(walls_only, output_path)

            print(f"Processed: {filename}")


if __name__ == '__main__':
    process_wall_extraction()
