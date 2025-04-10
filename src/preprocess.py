import cv2
import numpy as np
import os
import easyocr

reader = easyocr.Reader(['en'], gpu=False)

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

def remove_text_with_easyocr(img_color):
    """
    Use EasyOCR to remove detected text from a color image,
    but filter out false positives (e.g., thin walls).

    :param img_color: BGR image
    :return: Image with text removed
    """
    results = reader.readtext(img_color)

    for (bbox, text, conf) in results:
        pts = np.array(bbox).astype(np.int32)
        x_min = min(p[0] for p in pts)
        x_max = max(p[0] for p in pts)
        y_min = min(p[1] for p in pts)
        y_max = max(p[1] for p in pts)

        width = x_max - x_min
        height = y_max - y_min
        aspect_ratio = max(width / (height + 1e-5), height / (width + 1e-5))

        # filter 1 : ignore small areas
        if width < 3 or height < 3:
            continue

        # filter 2 : ignore the long line (because maybe it's a wall)
        if aspect_ratio > 20:
            continue

        # filter 3 : ignore low confidence
        if conf < 0.3:
            continue

        cv2.rectangle(img_color, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)

    return img_color



def preprocess_image(img, center_margin=330):
    """
    Preprocess the image: enhance contrast, remove noise, and convert to grayscale.

    :param img: Input color image (BGR)
    :return: Preprocessed grayscale image
    """
    img_no_text = remove_text_with_easyocr(img)
    # Convert to grayscale
    gray = cv2.cvtColor(img_no_text, cv2.COLOR_BGR2GRAY)
    # Enhance contrast
    gray_enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)

    # Create a mask for the center region
    height, width = gray.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    center_x_start = width // 2 - center_margin
    center_x_end = width // 2 + center_margin

    center_x_start = max(0, center_x_start)
    center_x_end = min(width, center_x_end)

    mask[:, center_x_start:center_x_end] = 255

    # Apply brightness correction to the center region
    center_region = gray_enhanced.copy()
    center_region = cv2.convertScaleAbs(center_region, alpha=1.5, beta=50)  # Increase brightness in the center

    # Combine the brightness-corrected center with the original image
    result = gray_enhanced.copy()
    result[mask == 255] = center_region[mask == 255]

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(result, (5, 5), 0)

    return blurred


def apply_threshold(img, center_margin=330, center_threshold=180, default_threshold=130):
    """
    Apply partial thresholding: higher threshold in the center, default threshold elsewhere.

    :param img: Grayscale image as a NumPy array
    :param center_margin: Margin in pixels to define the center region
    :param center_threshold: Threshold value for the center region
    :param default_threshold: Threshold value for the rest of the image
    :return: Binary thresholded image
    """
    height, width = img.shape

    # Create a mask for the center region
    mask = np.zeros((height, width), dtype=np.uint8)

    # Define the center region (considering the margin)
    center_x_start = width // 2 - center_margin
    center_x_end = width // 2 + center_margin

    # Ensure the center region stays within image bounds
    center_x_start = max(0, center_x_start)
    center_x_end = min(width, center_x_end)

    # Set the center region to 255 in the mask
    mask[:, center_x_start:center_x_end] = 255

    # Apply thresholding to the entire image with the default threshold
    _, binary_default = cv2.threshold(img, default_threshold, 255, cv2.THRESH_BINARY)

    # Apply thresholding to the center region with the higher threshold
    _, binary_center = cv2.threshold(img, center_threshold, 255, cv2.THRESH_BINARY)

    # Combine the results: use center threshold in the center, default elsewhere
    binary = binary_default.copy()
    binary[mask == 255] = binary_center[mask == 255]

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
