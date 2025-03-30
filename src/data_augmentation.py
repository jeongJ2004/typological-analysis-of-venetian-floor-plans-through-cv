import os
import cv2
import albumentations as A
from tqdm import tqdm

# ====== Config ======
input_dir = "patches_inputs"
label_dir = "patches_labels"
output_input_dir = "aug_inputs"
output_label_dir = "aug_labels"
os.makedirs(output_input_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# ====== Augmentation Pipeline ======
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),                # Horizontal flip
    A.VerticalFlip(p=0.5),                  # Vertical flip
    A.RandomRotate90(p=0.5),                # Random 90-degree rotation
], additional_targets={'label': 'image'})   # Treat label as an image for identical transformations

# ====== Run Augmentation ======
patch_files = sorted(os.listdir(input_dir))
label_files = sorted(os.listdir(label_dir))

assert len(patch_files) == len(label_files), "Mismatch between input and label count!"

aug_id = 0  # ID counter for saving augmented images

for img_name, label_name in tqdm(zip(patch_files, label_files), total=len(patch_files)):
    # Load input image and corresponding label
    img = cv2.imread(os.path.join(input_dir, img_name))
    label = cv2.imread(os.path.join(label_dir, label_name))

    # Save the original image and label (optional)
    cv2.imwrite(os.path.join(output_input_dir, f"patch_{aug_id:04d}.jpg"), img)
    cv2.imwrite(os.path.join(output_label_dir, f"patch_{aug_id:04d}.png"), label)
    aug_id += 1

    # Apply augmentations N times per image
    for i in range(3):  # 3x augmentation
        augmented = augmentations(image=img, label=label)
        aug_img = augmented["image"]
        aug_label = augmented["label"]

        # Save the augmented image and label
        cv2.imwrite(os.path.join(output_input_dir, f"patch_{aug_id:04d}.jpg"), aug_img)
        cv2.imwrite(os.path.join(output_label_dir, f"patch_{aug_id:04d}.png"), aug_label)
        aug_id += 1

print(f"Done!!!!!!!!!! Total of {aug_id} patches generated.")
