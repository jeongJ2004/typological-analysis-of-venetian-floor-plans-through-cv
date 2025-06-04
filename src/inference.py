# Import required libraries
import os
import torch # For PyTorch tensor operations and deep learning
import numpy as np
import cv2 # For image processing (e.g., saving images in BGR format)
from PIL import Image # For opening and converting images
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor # For Segformer model and preprocessing
from IPython.display import display, Image as ColabImage

# Set up paths for model directory, input image folder, and output results
MODEL_DIR = "/content/drive/MyDrive/segmentation_project/segformer-final(model)"
INPUT_FOLDER = "/content/drive/MyDrive/segmentation_project/thresholded-imgs-cut"
OUTPUT_FOLDER = "/content/drive/MyDrive/segmentation_project/segformer_results_final"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Ensure output folder exists

# Define key parameters
num_classes = 4  # Number of segmentation classes: background, wall, stairs, window
patch_size = 256  # Size of each square patch (in pixels)
stride = 128  # Step size for sliding window (overlap control)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Define visualization colors for each class (in RGB format)
class_colors = {
    0: [0, 0, 0],       # Background - Black
    1: [255, 0, 0],     # Wall - Red
    2: [0, 255, 0],     # Stairs - Green
    3: [0, 0, 255]      # Window - Blue
}

# Load pretrained SegFormer model and processor
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_DIR, # Load from specified directory
    num_labels=num_classes, # Set number of output classes
    local_files_only=True # Use local files instead of downloading
).to(device) # Move model to GPU/CPU
model.eval()  # Set the model to evaluation mode (no gradient updates)
processor = SegformerImageProcessor(do_reduce_labels=False)  # Keep full label set

# List all floor plan images in the input directory
image_files = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

# Inference loop over each image
for image_file in image_files:
    print(f"▶️ Processing: {image_file}")
    input_path = os.path.join(INPUT_FOLDER, image_file)
    output_path = os.path.join(OUTPUT_FOLDER, f"seg_{os.path.splitext(image_file)[0]}.png")

    # Load and convert image to RGB
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img)
    orig_H, orig_W = img_np.shape[:2]

    # Pad image to ensure compatibility with sliding window logic
    pad_H = ((orig_H - patch_size) // stride + 1) * stride + patch_size
    pad_W = ((orig_W - patch_size) // stride + 1) * stride + patch_size
    padded_img = np.zeros((pad_H, pad_W, 3), dtype=np.uint8)
    padded_img[:orig_H, :orig_W, :] = img_np

    # Initialize logit accumulation and overlap count map
    pred_mask_sum = np.zeros((pad_H, pad_W, num_classes), dtype=np.float32)
    count_map = np.zeros((pad_H, pad_W, 1), dtype=np.float32)

    # Perform patch-wise sliding window inference
    for y in range(0, pad_H - patch_size + 1, stride):
        for x in range(0, pad_W - patch_size + 1, stride):
            patch = padded_img[y:y+patch_size, x:x+patch_size]

            # Preprocess image patch
            inputs = processor(images=patch, return_tensors="pt", size=(256, 256))
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Perform model inference without gradient tracking
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # Raw prediction scores
                logits = torch.nn.functional.interpolate(
                    logits,
                    size=(patch_size, patch_size),
                    mode="bilinear",
                    align_corners=False
                )
                logits = logits[0].cpu().numpy()  # Convert to numpy array

            # Rearrange shape from [C, H, W] to [H, W, C]
            logits = np.transpose(logits, (1, 2, 0))
            pred_mask_sum[y:y+patch_size, x:x+patch_size] += logits
            count_map[y:y+patch_size, x:x+patch_size] += 1

    # Normalize logits by number of overlapping patches
    pred_mask_avg = pred_mask_sum / count_map
    pred_mask_avg = pred_mask_avg[:orig_H, :orig_W, :]  # Crop back to original size

    # Initialize final class mask with -1 (undefined)
    final_mask = np.full((orig_H, orig_W), -1, dtype=np.int32)

    # Extract logits for each class
    background_logits = pred_mask_avg[:, :, 0]
    wall_logits = pred_mask_avg[:, :, 1]
    stairs_logits = pred_mask_avg[:, :, 2]
    windows_logits = pred_mask_avg[:, :, 3]

    # Custom thresholding logic to better detect rare class (stairs)
    threshold_stairs = -3.7
    threshold_background = 3
    stairs_condition = (
        (stairs_logits > threshold_stairs)
        & (background_logits < threshold_background)
        & (stairs_logits > wall_logits)
        & (stairs_logits > windows_logits)
    )
    final_mask[stairs_condition] = 2  # Assign "stairs" class where condition holds

    # Apply argmax for remaining pixels
    remaining_pixels = (final_mask == -1)
    final_mask[remaining_pixels] = np.argmax(pred_mask_avg[remaining_pixels], axis=-1)

    # Create a color image for visualization
    color_mask = np.zeros((orig_H, orig_W, 3), dtype=np.uint8)
    for cls_id, color in class_colors.items():
        color_mask[final_mask == cls_id] = color

    # Save the output image with predicted segmentation overlay
    cv2.imwrite(output_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    print(f"Saved to: {output_path}")

print("Inference completed for all images !")

# -------------------------------------------------------------------------------------------------------------
# GT logits stat code :

# # Load the ground truth (GT) segmentation mask
# gt_path = "/content/drive/MyDrive/segmentation_project/gt038_cleaned.png"
# gt_img = cv2.imread(gt_path)  # Read the image in BGR format
# gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
#
# # Resize ground truth mask to match the model prediction resolution
# gt_img_resized = cv2.resize(
#     gt_img,
#     (pred_mask_avg.shape[1], pred_mask_avg.shape[0]),
#     interpolation=cv2.INTER_NEAREST
# )
#
# # Define the RGB-to-class mapping used in the ground truth
# COLOR_MAP = {
#     (0, 0, 0): 0,       # Background (black)
#     (255, 0, 0): 1,     # Wall (red)
#     (0, 255, 0): 2,     # Stairs (green)
#     (0, 0, 255): 3      # Window (blue)
# }
#
# # Initialize a blank 2D mask to store class indices
# gt_mask = np.zeros((gt_img_resized.shape[0], gt_img_resized.shape[1]), dtype=np.uint8)
#
# # Convert RGB colors in the ground truth to class indices
# for rgb, cls in COLOR_MAP.items():
#     match = np.all(gt_img_resized == np.array(rgb), axis=-1)
#     gt_mask[match] = cls
#
# # Define class names for readability
# class_names = ["Background", "Wall", "Stairs", "Window"]
#
# # Compute and display logit statistics for each class based on GT pixels
# for class_id in range(4):
#     mask = (gt_mask == class_id)  # Boolean mask for current class
#     class_logits = pred_mask_avg[:, :, class_id][mask]  # Extract logits for GT-matched pixels
#
#     print(f"\n Logit statistics for class: '{class_names[class_id]}' (based on GT mask)")
#     print(f"Pixels:  {class_logits.size}")
#     print(f"Mean:    {class_logits.mean():.6f}")
#     print(f"Median:  {np.median(class_logits):.6f}")
#     print(f"10%:     {np.percentile(class_logits, 10):.6f}")
#     print(f"90%:     {np.percentile(class_logits, 90):.6f}")


# ----------------------------------------------------------------------------------------------------------------

# Metrics code :

# from sklearn.metrics import precision_score, recall_score, f1_score
#
# def compute_all_class_metrics(gt_mask, pred_mask, class_names):
#     """
#     Compute precision, recall, and F1 score for each class in a multi-class segmentation task.
#
#     Args:
#         gt_mask (np.ndarray): Ground truth mask with class indices.
#         pred_mask (np.ndarray): Predicted segmentation mask with class indices.
#         class_names (list of str): List of class names corresponding to class indices.
#
#     Returns:
#         dict: A dictionary containing per-class metrics.
#     """
#     metrics = {}
#
#     for class_id, class_name in enumerate(class_names):
#         # Create binary masks for the current class (one-vs-all)
#         gt_binary = (gt_mask.flatten() == class_id).astype(np.uint8)
#         pred_binary = (pred_mask.flatten() == class_id).astype(np.uint8)
#
#         # Compute evaluation metrics
#         precision = precision_score(gt_binary, pred_binary, zero_division=0)
#         recall = recall_score(gt_binary, pred_binary, zero_division=0)
#         f1 = f1_score(gt_binary, pred_binary, zero_division=0)
#
#         # Print results
#         print(f"\n {class_name} (class_id = {class_id})")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall:    {recall:.4f}")
#         print(f"F1 Score:  {f1:.4f}")
#
#         # Store results in dictionary
#         metrics[class_name] = {
#             "precision": precision,
#             "recall": recall,
#             "f1": f1
#         }
#
#     return metrics
#
# # Execute metric evaluation
# class_names = ["Background", "Wall", "Stairs", "Window"]
# all_metrics = compute_all_class_metrics(gt_mask, pred_mask_class, class_names)


