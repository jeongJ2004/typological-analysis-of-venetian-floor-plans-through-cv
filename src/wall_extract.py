import cv2
import numpy as np


def load_image(img_path):
    """
    Load a preprocessed binary (thresholded) image.

    :param img_path: Path to the preprocessed image
    :return: NumPy array representing the binary image
    :raises ValueError: If the image cannot be loaded
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Could not load image: {img_path}')
    return img


def extract_walls(img):
    """
    Extracts walls from a binary image using improved edge detection and morphological operations.

    :param img: Input binary image (already thresholded)
    :return: Processed image with detected walls highlighted
    """
    # Step 1: Apply Adaptive Threshold to Enhance Wall Contrast
    _, binary = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)

    # Step 2: Apply Canny Edge Detection. (the lower threshold1 is, the better thin wall detection is. But it risks noise detection too)
    edges = cv2.Canny(binary, threshold1=40, threshold2=160)

    # Step 3: Morphological Operations
    kernel = np.ones((2, 2), np.uint8)  # Slightly stronger effect to capture thin walls
    dilated = cv2.dilate(edges, kernel, iterations=1)  # Increase dilation to connect weak edges
    eroded = cv2.erode(dilated, kernel, iterations=1)  # Reduce erosion to avoid losing structure

    # Step 4: Find Contours and Fill Walls
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wall_mask = np.zeros_like(img)
    cv2.drawContours(wall_mask, contours, -1, (255), thickness=5)

    # Step 5: Invert Colors (Make Walls Black, Background White)
    # wall_mask = cv2.bitwise_not(wall_mask) # this will invert the black to white and white to black

    return wall_mask


if __name__ == '__main__':
    image_path = "../results/thresholded_imgs/threshold_110.jpg"
    img = load_image(image_path)

    # Extract walls from the thresholded image
    wall_img = extract_walls(img)

    # Save and display the extracted walls
    output_path = image_path.replace(".jpg", "_walls.jpg")
    cv2.imwrite(output_path, wall_img)
    print(f"Saved wall extraction result to {output_path}")

    # Show the detected walls
    cv2.imshow("Extracted Walls", wall_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
