import os
import cv2

# Define directory paths
WALL_FOLDER = "/content/drive/MyDrive/segmentation_project/wall-extracted-imgs-cut"         # Folder containing OpenCV wall-extracted images
SEGFORMER_FOLDER = "/content/drive/MyDrive/segmentation_project/segformer_results_final"     # Folder containing SegFormer segmentation results
HYBRID_FOLDER = "/content/drive/MyDrive/segmentation_project/hybrid_results_final"           # Output folder for hybrid results

# Create output directory if it doesn't exist
os.makedirs(HYBRID_FOLDER, exist_ok=True)

# Retrieve and sort filenames of wall extraction images
wall_files = sorted([
    f for f in os.listdir(WALL_FOLDER)
    if f.lower().endswith(".jpg")
])

# Iterate over each wall extraction image
for wall_file in wall_files:
    wall_path = os.path.join(WALL_FOLDER, wall_file)

    # Generate corresponding SegFormer output filename
    base_name = os.path.splitext(wall_file)[0]  # Remove extension
    segformer_filename = f"seg_{base_name}.png"
    segformer_path = os.path.join(SEGFORMER_FOLDER, segformer_filename)

    # Skip if corresponding SegFormer output is missing
    if not os.path.exists(segformer_path):
        print(f"SegFormer output missing: {segformer_filename} (skipping)")
        continue

    # Load both wall and segmentation images
    wall_img = cv2.imread(wall_path)
    logit_img = cv2.imread(segformer_path)

    if wall_img is None or logit_img is None:
        print(f"Failed to load images: {wall_file}")
        continue

    # Align image sizes to the smaller of the two
    H = min(wall_img.shape[0], logit_img.shape[0])
    W = min(wall_img.shape[1], logit_img.shape[1])
    wall_img = wall_img[:H, :W]
    logit_img = logit_img[:H, :W]

    # Initialize hybrid image with wall-extracted base
    hybrid_img = wall_img.copy()

    # Detect stairs using green color range (in BGR format)
    green_mask = cv2.inRange(logit_img, (0, 200, 0), (100, 255, 100))  # BGR thresholds for green

    # Detect windows using blue color range (in BGR format)
    blue_mask = cv2.inRange(logit_img, (200, 0, 0), (255, 100, 100))  # BGR thresholds for blue

    # Overlay detected features on the wall image
    hybrid_img[green_mask > 0] = [0, 255, 0]   # Overwrite with bright green for stairs
    hybrid_img[blue_mask > 0] = [255, 0, 0]    # Overwrite with bright blue for windows

    # Save the hybrid image
    save_path = os.path.join(HYBRID_FOLDER, f"hybrid_{base_name}.png")
    cv2.imwrite(save_path, hybrid_img)
    print(f"Saved hybrid image to: {save_path}")

print("All hybrid images successfully generated!")
