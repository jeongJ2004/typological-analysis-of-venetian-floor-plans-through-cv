# Import necessary libraries
import torch  # For PyTorch tensor operations and deep learning
import numpy as np
import cv2  # For image processing (e.g., saving images in BGR format)
from PIL import Image  # For opening and converting images
from io import BytesIO  # For handling uploaded file data as bytes
# from google.colab import files  # For file upload functionality in Colab
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor  # For Segformer model and preprocessing
# from IPython.display import display, Image as ColabImage  # For displaying images in Colab

# Define constants
MODEL_DIR = "/content/drive/MyDrive/segmentation_project/segformer-final"  # Directory where the trained model is stored
RESULT_SAVE_PATH = "/content/drive/MyDrive/segmentation_project/segformer_result_full2.png"  # Path to save the inference result
num_classes = 4  # Number of segmentation classes (background, wall, stairs, window)
patch_size = 256  # Size of each image patch for sliding window inference
stride = 128  # Step size for sliding window (overlap between patches)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU

# Define color mapping for visualization
class_colors = {
    0: [0, 0, 0],       # Background (black)
    1: [255, 0, 0],     # Wall (red)
    2: [0, 255, 0],     # Stairs (green)
    3: [0, 0, 255]      # Window (blue)
}

# Load the trained model and processor
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_DIR,  # Load from specified directory
    num_labels=num_classes,  # Set number of output classes
    local_files_only=True  # Use local files instead of downloading
).to(device)  # Move model to GPU/CPU
model.eval()  # Set model to evaluation mode (no training)
processor = SegformerImageProcessor(do_reduce_labels=False)  # Initialize processor without reducing labels

# Upload an image for inference
print("Upload an image")  # Prompt user to upload an image

# This code will only work in Colab

uploaded = files.upload()  # Upload file via Colab interface
filename = list(uploaded.keys())[0]  # Get the filename of the uploaded image
img = Image.open(BytesIO(uploaded[filename])).convert("RGB")  # Open image and convert to RGB
img_np = np.array(img)  # Convert PIL image to NumPy array
orig_H, orig_W = img_np.shape[:2]  # Get original height and width of the image

# Pad the image to fit patch size and stride
pad_H = ((orig_H - patch_size) // stride + 1) * stride + patch_size  # Calculate padded height
pad_W = ((orig_W - patch_size) // stride + 1) * stride + patch_size  # Calculate padded width
padded_img = np.zeros((pad_H, pad_W, 3), dtype=np.uint8)  # Create a padded image array filled with zeros
padded_img[:orig_H, :orig_W, :] = img_np  # Copy original image into padded array

# Arrays to accumulate predictions and counts
pred_mask_sum = np.zeros((pad_H, pad_W, num_classes), dtype=np.float32)  # Array to sum logits for each class
count_map = np.zeros((pad_H, pad_W, 1), dtype=np.float32)  # Array to count overlaps for averaging

# Perform sliding window inference
for y in range(0, pad_H - patch_size + 1, stride):  # Loop over y-axis with stride
    for x in range(0, pad_W - patch_size + 1, stride):  # Loop over x-axis with stride
        patch = padded_img[y:y+patch_size, x:x+patch_size]  # Extract a patch from padded image

        # Preprocess the patch
        inputs = processor(images=patch, return_tensors="pt", size=(256, 256))  # Process patch for model input
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input tensors to device (GPU/CPU)

        # Run inference without gradient computation
        with torch.no_grad():
            outputs = model(**inputs)  # Forward pass through the model

            # Resize model output to patch size
            logits = outputs.logits  # Get raw prediction scores
            logits = torch.nn.functional.interpolate(
                logits,  # Interpolate logits
                size=(patch_size, patch_size),  # Resize to patch size
                mode="bilinear",  # Use bilinear interpolation
                align_corners=False  # No corner alignment for simplicity
            )
            logits = logits[0].cpu().numpy()  # Move to CPU and convert to NumPy array

        logits = np.transpose(logits, (1, 2, 0))  # Reshape from [C, H, W] to [H, W, C]

        # Accumulate predictions and counts
        pred_mask_sum[y:y+patch_size, x:x+patch_size] += logits  # Add logits to sum array
        count_map[y:y+patch_size, x:x+patch_size] += 1  # Increment count for averaging

# Compute average prediction and crop to original size
pred_mask_avg = pred_mask_sum / count_map  # Average the accumulated logits
pred_mask_avg = pred_mask_avg[:orig_H, :orig_W, :]  # Crop to original image size
final_mask = np.argmax(pred_mask_avg, axis=-1)  # Get class with highest score per pixel

# Create colored mask for visualization
color_mask = np.zeros((orig_H, orig_W, 3), dtype=np.uint8)  # Empty RGB array for colored mask
for cls_id, color in class_colors.items():  # Assign colors based on class IDs
    color_mask[final_mask == cls_id] = color  # Set pixels to corresponding class color

# Save and display the result
cv2.imwrite(RESULT_SAVE_PATH, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))  # Save image in BGR format
print(f"Done ! saved image: {RESULT_SAVE_PATH}")  # Confirm completion and save path
display(ColabImage(RESULT_SAVE_PATH))  # Display the result in Colab