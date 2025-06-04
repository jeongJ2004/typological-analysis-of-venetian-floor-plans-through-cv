# This is for Google colab to call the contents in drive

# from google.colab import files
# uploaded = files.upload()  # upload the zip file
#
# import zipfile
import os

#
# with zipfile.ZipFile("project_dataset2.zip", 'r') as zip_ref:
#     zip_ref.extractall("dataset")  # dataset/aug_inputs, dataset/aug_labels generated


# import os
# os.environ["WANDB_DISABLED"] = "true"
#
# from google.colab import drive
# drive.mount('/content/drive')


"""
NOTE:
This project uses the pretrained model 'nvidia/segformer-b4-finetuned-ade-512-512' under the NVIDIA Source Code License.
This license allows use for non-commercial research or evaluation purposes only.
See: https://github.com/NVIDIA/semantic-segmentation/blob/main/LICENSE
"""
# ALWAYS important to write the reference/license !

# Path where the model is gonna be saved
MODEL_SAVE_DIR = "/content/drive/MyDrive/segmentation_project/segformer-final"

# 1. Import Libraries
import torch  # Import PyTorch for tensor operations and deep learning
import numpy as np
import cv2  # Import OpenCV for image processing (e.g., reading and converting images)
import torch.nn as nn  # for defining loss functions
from torch.utils.data import Dataset  # to create a custom dataset for training
from sklearn.model_selection import train_test_split  # to split data into training and validation sets
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer
)

# 2. Define Dataset Paths and Classes
data_dir = "/content/drive/MyDrive/segmentation_project/dataset/project_dataset2"  # Base directory for dataset
img_dir = os.path.join(data_dir, "aug_inputs")  # Directory for augmented input images
label_dir = os.path.join(data_dir, "aug_labels")  # Directory for augmented label masks
num_classes = 4  # Number of classes in the segmentation task (background, wall, stairs, window)

# Define a color map to map RGB values in masks to class indices
COLOR_MAP = {
    (0, 0, 0): 0,  # Background class (black)
    (255, 0, 0): 1,  # Wall class (red)
    (0, 255, 0): 2,  # Stairs class (green)
    (0, 0, 255): 3  # Window class (blue)
}


# 3. Function to Convert RGB Masks to Class Indices
def rgb_to_class(mask):
    h, w, _ = mask.shape  # Get height, width of the mask (ignoring channel dimension)
    class_mask = np.zeros((h, w), dtype=np.uint8)  # Create an empty array for class indices with same height and width
    unmatched = np.ones((h, w), dtype=bool)  # Create a boolean mask to track pixels not yet matched to a class

    # Iterate through each RGB color and its corresponding class in COLOR_MAP
    for rgb, cls in COLOR_MAP.items():
        match = np.all(mask == np.array(rgb), axis=-1)  # Check where mask matches the RGB value across all channels
        class_mask[match] = cls  # Assign the class index to matching pixels
        unmatched &= ~match  # Update unmatched mask by excluding matched pixels

    # Warn if any pixels don't match defined RGB values
    if np.any(unmatched):
        print("Undefined RGB values detected :", np.unique(mask[unmatched], axis=0))
        class_mask[unmatched] = 0  # Assign unmatched pixels to background class (0)

    return class_mask


# 4. Custom Dataset Class for Floor Plan Segmentation
class FloorPlanDataset(Dataset):
    def __init__(self, image_paths, label_paths, processor):
        self.image_paths = image_paths  # List of paths to input images
        self.label_paths = label_paths  # List of paths to label masks
        self.processor = processor  # Segformer processor for image preprocessing

    def __len__(self):
        return len(self.image_paths)  # Return the total number of samples in the dataset

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])  # Load the image from file
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image from BGR (OpenCV default) to RGB

        label = cv2.imread(self.label_paths[idx])  # Load the label mask from file
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)  # Convert label from BGR to RGB
        label = rgb_to_class(label)  # Convert RGB mask to class indices
        label = np.clip(label, 0, num_classes - 1)  # Ensure class indices are within valid range (0 to 3) => For ex, if I have [-3,0,1,5,8], it will become [0,0,1,3,3]

        # Process image and label for Segformer model input
        encoding = self.processor(
            image,  # Input image
            return_tensors="pt", # Return PyTorch tensors
            size=(256, 256), # Resize image to 256x256
            do_reduce_labels=False # Keep original label values (no reduction)
        )
        encoding["labels"] = torch.tensor(label, dtype=torch.long)  # Add label tensor to encoding
        return {k: v.squeeze(0) for k, v in encoding.items()}  # Remove batch dimension from tensors


# 5. Prepare Dataset
# Get sorted lists of image and label file paths
image_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")])
label_files = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".png")])

# Split dataset into training and validation sets (80% train, 20% validation)
train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    image_files, label_files, test_size=0.2, random_state=42
) # Ensures reproducible splits.

processor = SegformerImageProcessor(do_reduce_labels=False)  # Initialize Segformer processor without label reduction
train_dataset = FloorPlanDataset(train_imgs, train_labels, processor)  # Create training dataset
val_dataset = FloorPlanDataset(val_imgs, val_labels, processor)  # Create validation dataset

# 6. Load Model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b4-finetuned-ade-512-512",  # Load pretrained Segformer model
    num_labels=num_classes,  # Set number of output classes to 4
    ignore_mismatched_sizes=True  # Ignore size mismatches between pretrained and new head
)


# 7. Define Evaluation Metrics
def compute_metrics(eval_preds):
    import torch.nn.functional as F  # Import functional module for interpolation
    logits, labels = eval_preds  # Get model predictions (logits) and ground truth labels
    preds = np.argmax(logits, axis=1)  # Convert logits to class predictions by taking argmax

    # Resize predictions to match label size if needed
    if preds.shape != labels.shape:
        preds = torch.from_numpy(preds).unsqueeze(1).float()  # Convert to tensor and add channel dimension
        preds = F.interpolate(preds, size=labels.shape[1:], mode="nearest")  # Resize to label size
        preds = preds.squeeze(1).long().numpy()  # Remove channel dimension and convert back to numpy

    acc = (preds == labels).mean()  # Calculate pixel-wise accuracy
    return {"overall_accuracy": acc}  # Return accuracy as a dictionary


# Custom Trainer with Weighted Loss
class WeightedLossTrainer(Trainer):
    # Inspired by https://medium.com/@MUmarAmanat/qlora-fine-tuning-of-llama-3-8b-on-aws-sagemaker-2a6e787d726b
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # Remove labels from inputs and store them
        outputs = model(**inputs)  # Forward pass through the model
        logits = outputs.logits  # Get raw predictions (logits)

        # Define class weights to balance loss (higher weights for rare classes)
        weights = torch.tensor([0.2, 1.0, 60.0, 50.0]).to(logits.device)

        # Resize labels to match logits size (64x64 from 256x256)
        labels_resized = torch.nn.functional.interpolate(
            labels.unsqueeze(1).float(),  # Add channel dimension and convert to float
            size=logits.shape[2:],  # Match logits spatial dimensions
            mode="nearest"  # Use nearest neighbor interpolation
        ).squeeze(1).long()  # Remove channel dimension and convert back to long

        loss_fct = nn.CrossEntropyLoss(weight=weights)  # Define cross-entropy loss with weights
        loss = loss_fct(logits, labels_resized)  # Compute weighted loss
        return (loss, outputs) if return_outputs else loss  # Return loss and outputs if requested


# 9. Set Training Parameters
training_args = TrainingArguments(
    output_dir="segformer-checkpoints",  # Directory to save checkpoints
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=2,  # Batch size for evaluation
    num_train_epochs=50,  # Number of training epochs
    eval_strategy="epoch",  # Evaluate after each epoch
    save_strategy="epoch",  # Save model after each epoch
    logging_strategy="epoch",  # Log metrics after each epoch
    learning_rate=5e-5,  # Learning rate for optimization
    save_total_limit=2,  # Keep only the last 2 checkpoints
    remove_unused_columns=False,  # Keep all dataset columns
    load_best_model_at_end=True,  # Load the best model based on metric at the end
    metric_for_best_model="overall_accuracy",  # Use accuracy to determine best model
    report_to="none"  # Disable reporting to external services (e.g., wandb)
)

# 10. Define Trainer and Start Training
trainer = WeightedLossTrainer(
    model=model,  # Pass the Segformer model
    args=training_args,  # Pass training arguments
    train_dataset=train_dataset,  # Pass training dataset
    eval_dataset=val_dataset,  # Pass validation dataset
    compute_metrics=compute_metrics  # Pass evaluation metrics function
)

trainer.train()  # Start the training process

# 11. Save Model
trainer.save_model(MODEL_SAVE_DIR)  # Save the trained model to specified directory
print(f"Training complete and model saved to Google Drive: {MODEL_SAVE_DIR}")  # Confirm training completion
