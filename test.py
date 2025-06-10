import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys

# --- 1. SETUP & CONFIGURATION ---
print("--- Actor Classifier Inference Script ---")

# --- Paths for Android ---
# Make sure these paths are correct for your device.
BASE_PATH = "/storage/emulated/0"
DOWNLOAD_DIR = os.path.join(BASE_PATH, "Download")
TEST_DATA_DIR = os.path.join(BASE_PATH, "Test") # <--- FOLDER FOR YOUR TEST IMAGES
MODEL_PATH = os.path.join(DOWNLOAD_DIR, "mohanlal_vs_mammootty_classifier.pth")

# --- Model parameters (MUST MATCH a training script) ---
NUM_CLASSES = 2
IMG_SIZE = 224 # Image size required by the MobileNetV2 model

# --- Class Names ---
# IMPORTANT: This order must match the folders ImageFolder used during training.
# ImageFolder sorts folders alphabetically, so 'Mammootty' would be 0 and 'Mohanlal' would be 1.
# We map the output to the desired names.
CLASS_NAMES = ['മമ്മൂട്ടി', 'Mohanlal']

# Set device (will be 'cpu' on most phones)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. VERIFY FILES AND FOLDERS ---
print("\n--- Step 1: Verifying Files & Folders ---")

# Check if the saved model file exists
if not os.path.isfile(MODEL_PATH):
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    print("Please make sure you have trained the model and it was saved correctly.")
    sys.exit()
print(f"Model file found at: {MODEL_PATH}")

# Check if the Test directory exists. If not, create it and give instructions.
if not os.path.isdir(TEST_DATA_DIR):
    print(f"Info: Test directory not found at '{TEST_DATA_DIR}'.")
    print("Creating the folder for you...")
    os.makedirs(TEST_DATA_DIR)
    print(f"\nPlease place the images you want to test inside the '{TEST_DATA_DIR}' folder and run this script again.")
    sys.exit()
print(f"Test image folder found at: {TEST_DATA_DIR}")


# --- 3. LOAD THE MODEL ARCHITECTURE & WEIGHTS ---
print("\n--- Step 2: Loading Model ---")

# 1. Re-create the exact model architecture from the training script
model = models.mobilenet_v2(weights=None) # Use weights=None as we will load our own

# 2. Get the number of input features for the classifier
num_ftrs = model.classifier[1].in_features

# 3. Replace the final layer to match the trained model
model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
print("Model architecture re-created successfully.")

# 4. Load the saved weights (the state_dict) into the model structure
# map_location=device is crucial for loading a model trained on a GPU onto a CPU
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
print("Trained weights loaded successfully.")

# 5. Set the model to evaluation mode
# This is critical! It disables layers like Dropout used only during training.
model.eval()
model = model.to(device)


# --- 4. DEFINE IMAGE TRANSFORMATIONS ---
print("\n--- Step 3: Defining Image Preprocessing ---")

# These transformations MUST match the 'val' transforms from your training script
# This ensures the test images are prepared in the exact same way as the validation images.
inference_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print("Image transformations are ready.")


# --- 5. RUN INFERENCE ON TEST IMAGES ---
print("\n--- Step 4: Running Predictions ---")

# Get a list of all valid image files in the test directory
try:
    image_files = [f for f in os.listdir(TEST_DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No image files (.jpg, .png, .jpeg) found in '{TEST_DATA_DIR}'.")
        print("Please add images to the folder and run again.")
        sys.exit()
except FileNotFoundError:
    print(f"Error: The test directory '{TEST_DATA_DIR}' was not found.")
    sys.exit()

# Loop through each image, preprocess it, and get a prediction
for image_name in image_files:
    try:
        image_path = os.path.join(TEST_DATA_DIR, image_name)

        # Open the image using Pillow (PIL)
        # .convert('RGB') handles images with an alpha channel (RGBA) or grayscale
        image = Image.open(image_path).convert('RGB')

        # Apply the transformations
        input_tensor = inference_transform(image)

        # Add a batch dimension (models expect a batch of images)
        # [C, H, W] -> [1, C, H, W]
        input_batch = input_tensor.unsqueeze(0)

        # Move the tensor to the correct device
        input_batch = input_batch.to(device)

        # Perform inference
        with torch.no_grad(): # No need to calculate gradients for testing
            output = model(input_batch)

        # Get the index of the highest score (the predicted class)
        _, predicted_idx = torch.max(output, 1)

        # Map the index to the class name
        # .item() gets the integer value from the tensor
        prediction = CLASS_NAMES[predicted_idx.item()]

        # Print the result
        print(f"File: {image_name:<30} --->   Prediction: {prediction}")

    except Exception as e:
        print(f"Could not process file {image_name}. Error: {e}")

print("\n--- Script Finished ---")
