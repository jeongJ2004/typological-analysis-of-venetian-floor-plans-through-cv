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
    h_kernel = np.ones((1, 4), np.uint8)
    v_kernel = np.ones((4, 1), np.uint8)
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # First, perform opening to remove small elements (text, dots, etc.)
    opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, small_kernel, iterations=1)

    # Remove thin horizontal and vertical lines (often text or small details)
    h_opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_opened = cv2.morphologyEx(h_opened, cv2.MORPH_OPEN, v_kernel, iterations=1)

    # Apply area filtering to remove small connected components (likely text)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(v_opened, connectivity=8)

    # Create output image
    filtered = np.zeros_like(v_opened)

    # Set minimum area threshold
    min_area = 30

    # Process each connected component
    for i in range(1, num_labels):  # Start from 1 to skip background
        area = stats[i, cv2.CC_STAT_AREA]
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

def process_single_wall_extraction(input_image_path, output_folder):
    """
    Process a single image to extract walls and remove text/numbers.

    :param input_image_path: Path to the input image
    :param output_folder: Folder to save the processed image
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Extract the filename from the input path
    filename = os.path.basename(input_image_path)

    # Load the image
    binary_img = load_image(input_image_path)

    # Convert to grayscale if it's not already
    if len(binary_img.shape) > 2:
        binary_img = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)

    # Ensure binary image (0 or 255)
    _, binary_img = cv2.threshold(binary_img, 128, 255, cv2.THRESH_BINARY)

    # Remove text and numbers
    walls_only = remove_text_and_numbers(binary_img)

    # Save the result
    output_path = os.path.join(output_folder, filename)
    save_result(walls_only, output_path)

    print(f"Processed: {filename}")

if __name__ == '__main__':
    # Specify the path to the single image you want to process
    input_image_path = "../results/thresholded-imgs-cut/img0131.jpg"
    output_folder = "../results/wall-extracted-imgs-cut"
    process_single_wall_extraction(input_image_path, output_folder)