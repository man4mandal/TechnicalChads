import os
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision.io import read_image
from torchvision.transforms.functional import resize, normalize

# --- Configuration ---
train_folder = r"E:\Hackathon\Processed_Dataset\train"
test_folder = r"E:\Hackathon\test_batch"
model_path = r"E:\Hackathon\best_facenet_model.pth"
threshold = 0.7  # similarity threshold to decide identity
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load FaceNet Model ---
print("🔧 Initializing FaceNet model...")
state_dict = torch.load(model_path, map_location=device)
state_dict = {k: v for k, v in state_dict.items() if not k.startswith('logits.')}

model = InceptionResnetV1(classify=False).to(device)
model.load_state_dict(state_dict, strict=False)
model.eval()
print("✅ Model loaded and ready.\n")

# --- Helper: Convert image to embedding ---
def get_embedding(image_path):
    img = read_image(image_path).float() / 255.0
    img = resize(img, [160, 160])
    img = normalize(img, mean=[0.5]*3, std=[0.5]*3)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        return model(img).squeeze(0)

# --- Helper: Cosine similarity metric ---
def cosine_similarity(a, b):
    a = a / a.norm()
    b = b / b.norm()
    return torch.dot(a, b).item()

# --- Step 1: Load embeddings of known people ---
print(f"📁 Scanning known identities in: {train_folder}\n")
known_embeddings = {}

for name in os.listdir(train_folder):
    person_dir = os.path.join(train_folder, name)
    if os.path.isdir(person_dir):
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png'))]
        if image_files:
            embedding = get_embedding(os.path.join(person_dir, image_files[0]))
            known_embeddings[name] = embedding
            print(f"🧠 Added reference for: {name}")

if not known_embeddings:
    print("⚠ No known faces found in training folder. Exiting.")
    exit()

# --- Step 2: Predict identities for test images ---
print(f"\n🖼  Starting face recognition on test images in: {test_folder}\n")

for file in os.listdir(test_folder):
    if file.lower().endswith(('.jpg', '.png')):
        image_path = os.path.join(test_folder, file)
        test_embedding = get_embedding(image_path)

        best_match = None
        highest_score = -1

        for name, ref_embedding in known_embeddings.items():
            score = cosine_similarity(test_embedding, ref_embedding)
            if score > highest_score:
                highest_score = score
                best_match = name

        matched_name = best_match if highest_score >= threshold else "Unknown"
        print(f"📸 {file} → {matched_name} (score: {highest_score:.4f})")

print("\n🏁 All done! Recognition completed.")
