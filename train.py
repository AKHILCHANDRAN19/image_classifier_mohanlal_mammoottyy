import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import os
import time
from tqdm import tqdm
import sys

# --- 1. SETUP & CONFIGURATION ---
print("--- Actor Classifier Training Script Initialized ---")

# --- Paths for Android ---
BASE_PATH = "/storage/emulated/0"
DATA_DIR = os.path.join(BASE_PATH, "Datasets")
DOWNLOAD_DIR = os.path.join(BASE_PATH, "Download")
MODEL_SAVE_PATH = os.path.join(DOWNLOAD_DIR, "mohanlal_vs_mammootty_classifier.pth")

# --- Model parameters ---
NUM_CLASSES = 2  # Mohanlal vs. Mammootty
BATCH_SIZE = 32  # Reduce this to 16 or 8 if you get "Out of Memory" errors
NUM_EPOCHS = 10  # 10 is a good starting point. Increase for potentially better accuracy.
LEARNING_RATE = 0.001
IMG_SIZE = 224 # Image size required by the MobileNetV2 model

# Check if a GPU is available (unlikely on mobile, but it's good practice)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. DATA VERIFICATION & PREPARATION ---
print("\n--- Step 1: Verifying Data ---")

# Ensure the main Datasets and Download directories exist
if not os.path.isdir(DATA_DIR):
    print(f"Error: The directory '{DATA_DIR}' was not found. Please create it and place your actor folders inside.")
    sys.exit() # Exit the script if the main data folder is missing

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# torchvision.datasets.ImageFolder will automatically find class folders.
# We just need to verify they exist for user feedback.
try:
    class_folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if len(class_folders) < 2:
        print(f"Error: Expected at least 2 subfolders in '{DATA_DIR}', but found {len(class_folders)}.")
        print("Please make sure you have both 'Mohanlal' and 'Mammootty' folders inside 'Datasets'.")
        sys.exit()
    print(f"Found class folders: {class_folders}")
except FileNotFoundError:
    print(f"Error: The directory '{DATA_DIR}' was not found.")
    sys.exit()


# --- 3. CREATE DATASET & DATALOADERS ---
print("\n--- Step 2: Creating Datasets and Dataloaders ---")

# Define transformations for the images
# Data augmentation (random crops, flips) is crucial for the training set to help the model generalize
# The validation set only needs basic resizing and normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load the full dataset using ImageFolder. This is the magic part!
# It automatically finds images and assigns labels from the folder names.
full_dataset = datasets.ImageFolder(DATA_DIR)

# Split the dataset into training and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# IMPORTANT: Apply the correct transforms to each split after splitting
train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']

# Create DataLoaders to feed data to the model in batches
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = full_dataset.classes

print(f"Classes discovered: {class_names} (0: {class_names[0]}, 1: {class_names[1]})")
print(f"Total images: {len(full_dataset)}")
print(f"Training set size: {dataset_sizes['train']}")
print(f"Validation set size: {dataset_sizes['val']}")


# --- 4. DEFINE THE MODEL (Transfer Learning) ---
print("\n--- Step 3: Defining the Model ---")

# Load a pre-trained MobileNetV2 model, which has learned features from millions of images
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze all the existing layers in the model
for param in model.parameters():
    param.requires_grad = False

# Get the number of input features for the final layer
num_ftrs = model.classifier[1].in_features

# Replace the final layer with a new one that matches our number of classes (2)
# The parameters of this new layer are automatically set to `requires_grad=True`
model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(device)
print("Model loaded and final layer customized for our task.")

# --- Define the Loss Function and Optimizer ---
criterion = nn.CrossEntropyLoss()
# We only want to train the parameters of our new, final layer
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)


# --- 5. TRAINING LOOP ---
print("\n--- Step 4: Starting Model Training ---")
print(f"Training for {NUM_EPOCHS} epochs. This will be slow on a CPU. Please be patient.")

start_time = time.time()
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Use tqdm for a nice progress bar
        progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase")
        
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward pass + optimize only in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Update the progress bar description with the current batch loss
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Save the model if it has the best validation accuracy so far
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            print(f"New best validation accuracy: {best_acc:.4f}. Saving model state.")
            # We save the model's state_dict, which is the efficient and recommended way.
            torch.save(model.state_dict(), MODEL_SAVE_PATH)


time_elapsed = time.time() - start_time
print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Best Validation Accuracy: {best_acc:4f}')


# --- 6. FINAL SAVE ---
print("\n--- Step 5: Finalizing ---")
print(f"The best performing model has been saved to: {MODEL_SAVE_PATH}")
print("\n--- Script Finished ---")
