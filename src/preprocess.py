import cv2
import numpy as np
import os


def load_image(img_path):
    """
        Load an image in grayscale mode.

        :param img_path: Path to the image file
        :return: Grayscale image as a NumPy array
        :raises ValueError: If the image can't be loaded
    """

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f'Could not load image: {img_path}')
    return img


def remove_rivers(img):
    """
    Remove blue rivers using color masking.

    :param img: Input color image (BGR)
    :return: Image with rivers removed
    """
    # Convert to HSV for better color segmentation -> Hue Saturation Value
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define blue color range for rivers
    lower_blue = np.array([100, 30, 30])
    upper_blue = np.array([130, 255, 255])

    # Create a mask for blue areas
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Dilate the mask to ensure complete removal of river areas
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Replace blue areas with white (background)
    img[mask == 255] = [255, 255, 255]

    return img


def preprocess_image(img):
    """
    Preprocess the image: enhance contrast, remove noise, and convert to grayscale.

    :param img: Input color image (BGR)
    :return: Preprocessed grayscale image
    """
    # Enhance contrast
    img_enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=0)

    # Convert to grayscale
    gray = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return blurred


def apply_threshold(img, threshold_val=130):
    """
    Apply binary thresholding to an image with Otsu's method as an option.

    :param img: Grayscale image as a NumPy array
    :param threshold_val: Threshold value (0-255), default is 150
    :return: Binary thresholded image
    """
    # Use Otsu's method for automatic thresholding if threshold_val is 0
    if threshold_val == 0:
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)

    # Morphological opening to remove small noise (text, artifacts)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary


def resize_image(img, max_width=1000, max_height=1000):
    """
        Resize an image while maintaining the aspect ratio to fit within the given dimensions.

        :param img: Input image as a NumPy array
        :param max_width: Maximum display width (default: 800 pixels)
        :param max_height: Maximum display height (default: 600 pixels)
        :return: Resized image
    """
    height, width = img.shape[:2]

    # Resize only if the image is larger than the max dimensions
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_size = (int(width * scale), int(height * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    return img


def save_result(img, output_path):
    cv2.imwrite(output_path, img)
    print(f"Saved image to {output_path}")


def process_image():
    input_folder = "../data"
    thresholded_folder = "../results/thresholded-imgs"
    wall_extracted_folder = "../results/wall-extracted-imgs"
    os.makedirs(thresholded_folder, exist_ok=True)
    os.makedirs(wall_extracted_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            img = load_image(img_path)

            #Step 1 : Remove rivers
            img_no_rivers = remove_rivers(img)

            #Step 2 : Preprocess image
            img_preprocessed = preprocess_image(img_no_rivers)

            #Step 3 : Apply thresholding
            binary_img = apply_threshold(img_preprocessed)

            #Step 4 : Save the thresholded images
            thresholded_path = os.path.join(thresholded_folder, filename)
            save_result(binary_img, thresholded_path)

            print(f"Processed: {filename}")


if __name__ == '__main__':
    process_image()
