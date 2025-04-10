import cv2
import numpy as np
import os
from scipy.spatial import KDTree

def clean_gt_image_kdtree(image_path, save_path):
    """
    Cleans a GT image by replacing fuzzy colors with the nearest class color (Red, Green, Blue),
    while preserving black pixels (background).
    """
    # Load and convert to RGB
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Define class colors
    RED = [255, 0, 0]    # Wall
    GREEN = [0, 255, 0]  # Stairs
    BLUE = [0, 0, 255]   # Window
    BLACK = [0, 0, 0]    # Background

    # KDTree with RGB class colors (excluding black)
    valid_colors = np.array([RED, GREEN, BLUE])
    color_tree = KDTree(valid_colors)

    # Flatten the image for processing
    h, w = img_rgb.shape[:2]
    reshaped = img_rgb.reshape((-1, 3))

    # For each pixel, find the nearest color from valid_classes then reshape back to the original image shape
    _, nearest_indices = color_tree.query(reshaped)
    corrected_colors = valid_colors[nearest_indices].reshape((h, w, 3))

    # Preserve original black pixels
    black_mask = np.all(img_rgb == BLACK, axis=-1)
    corrected_colors[black_mask] = BLACK

    # Convert to BGR and save
    corrected_bgr = cv2.cvtColor(corrected_colors.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, corrected_bgr)
    print(f"Cleaned GT saved to: {save_path}")


def extract_patches(input_img_path, label_img_path,
                    out_input_dir, out_label_dir,
                    resize_width, resize_height,
                    patch_size=512, stride=256):
    """
    Extracts patches from input and GT images after resizing.
    Saves RGB patches for both input and label images.
    """
    # Create output directories if they don't exist
    os.makedirs(out_input_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    # Load the input image and GT label image
    input_img = cv2.imread(input_img_path)
    label_img = cv2.imread(label_img_path)

    # Check if the images were loaded successfully
    if input_img is None or label_img is None:
        raise ValueError("Error: One or both images could not be loaded.")

    # Resize both input and label image to the same size (fixed)
    input_resized = cv2.resize(input_img, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    label_resized = cv2.resize(label_img, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST)

    # Extract patches using sliding window
    patch_id = 0  # Initialize patch counter
    for y in range(0, resize_height - patch_size + 1, stride):  # Loop over height with stride
        for x in range(0, resize_width - patch_size + 1, stride):  # Loop over width with stride
            # Extract patches from input and label images
            input_patch = input_resized[y:y+patch_size, x:x+patch_size]  # Extract input patch
            label_patch = label_resized[y:y+patch_size, x:x+patch_size]  # Extract label patch

            # Skip patches that are smaller than the defined patch size (at the image boundaries)
            if input_patch.shape[0] != patch_size or input_patch.shape[1] != patch_size:
                continue
            if label_patch.shape[0] != patch_size or label_patch.shape[1] != patch_size:
                continue

            # Define file paths for saving the patches
            input_patch_path = os.path.join(out_input_dir, f"patch_{patch_id:04d}.jpg")   # Path for input patch
            label_patch_path = os.path.join(out_label_dir, f"patch_{patch_id:04d}.png")   # Path for label patch

            # Save the patches to disk
            cv2.imwrite(input_patch_path, input_patch)
            cv2.imwrite(label_patch_path, label_patch)

            patch_id += 1  # Increment patch counter

    # Print the total number of saved patches
    print(f"Total {patch_id} patches saved.")


# ====== USAGE ======

# Step 1: Clean GT
clean_gt_image_kdtree(
    image_path="../data/gt038.png",
    save_path="../data/gt038_cleaned.png"
)

# Step 2: Extract patches from cleaned GT
extract_patches(
    input_img_path="../data/img038.jpg",
    label_img_path="../data/gt038_cleaned.png",
    out_input_dir="patches_inputs",
    out_label_dir="patches_labels",
    resize_width=2707,
    resize_height=1920,
    patch_size=256,
    stride=128
)

