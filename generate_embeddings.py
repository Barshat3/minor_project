import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import pickle

# Dataset and output paths
DATASET_PATH = "/home/barshat/Desktop/minor_project/lfw"  # Update with your dataset path
OUTPUT_FILE = "embeddings.pickle"   # File to save embeddings

# Initialize the pre-trained FaceNet model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Preprocessing transformation for the images
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 160)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Function to process images and generate embeddings
def generate_embeddings(dataset_path, existing_embeddings=None):
    if existing_embeddings is None:
        existing_embeddings = {}

    embeddings = existing_embeddings  # Start with existing embeddings
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            if person_name in embeddings:
                print(f"Skipping {person_name}, embeddings already exist.")
                continue  # Skip if already processed
            print(f"Processing {person_name}...")
            embeddings[person_name] = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    # Preprocess the image
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = preprocess(img_rgb).unsqueeze(0)  # Add batch dimension

                    # Generate embedding
                    with torch.no_grad():
                        embedding = model(img_tensor).squeeze().numpy()  # Get embedding as numpy array
                    embeddings[person_name].append(embedding)
    return embeddings

# Load existing embeddings if available
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "rb") as f:
        existing_embeddings = pickle.load(f)
else:
    existing_embeddings = {}

# Generate embeddings and save to the file
embeddings = generate_embeddings(DATASET_PATH, existing_embeddings)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(embeddings, f)

print(f"Updated embeddings saved to {OUTPUT_FILE}")
