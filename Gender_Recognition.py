import torch
from torchvision import models, transforms
from torchvision.io import read_image
import os

# --- Model Setup ---
# Start with a ResNet-50 backbone, commonly used for classification tasks
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # We're dealing with two classes: Male and Female

# Load pre-trained weights for gender classification
model.load_state_dict(torch.load("best_gender_model.pth", map_location=torch.device("cpu")))
model.eval()

# --- Image Preprocessing ---
# Define the preprocessing pipeline: resize, normalize, etc.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Directory Scanning ---
test_dir = "test_batch"
image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print("No valid image files found in the folder.")
else:
    print(f"Found {len(image_files)} images. Starting gender prediction...\n")

# --- Prediction Loop ---
for idx, file_name in enumerate(image_files, 1):
    image_path = os.path.join(test_dir, file_name)
    try:
        image = read_image(image_path)
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()

        label = "Male" if prediction == 1 else "Female"
        print(f"[{idx}] {file_name} → Predicted Gender: {label}")

    except Exception as e:
        print(f"[{idx}] {file_name} → Error: {str(e)}")

print("\nPrediction complete.")