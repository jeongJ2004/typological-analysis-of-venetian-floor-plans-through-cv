import cv2


def load_image(img_path):
    """
        Load an image in grayscale mode.

        :param img_path: Path to the image file
        :return: Grayscale image as a NumPy array
        :raises ValueError: If the image can't be loaded
    """

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError('Could not load image: {}'.format(img_path))
    return img


def apply_threshold(img, threshold_val=150):
    """
        Apply binary thresholding to an image.

        :param img: Grayscale image as a NumPy array
        :param threshold_val: Threshold value (0-255), default is 150 but user can input its own desired value when user runs preprocess.py
        :return: Binary thresholded image
    """
    _, binary = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)
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


if __name__ == '__main__':
    image_path = "../data/img041.jpg"
    img = load_image(image_path)

    while True:
        threshold_val = int(input('Threshold value: '))

        if threshold_val == -1:
            print("Exit")
            break

        binary_img = apply_threshold(img, threshold_val)

        resized_img = resize_image(binary_img) # It's not necessary to do this but I prefer for when the cv window appears

        cv2.imshow(f"Threshold: {threshold_val}", resized_img)
        key = cv2.waitKey(0)

        if key == ord('s'):
            output_path = f"../results/thresholded_imgs/threshold2_{threshold_val}.jpg"
            cv2.imwrite(output_path, binary_img)
            print(f"Saved image to {output_path}")

        cv2.destroyAllWindows()
