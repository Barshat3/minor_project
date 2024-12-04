import torch
from torchvision import transforms
import numpy as np

# Preprocessing transformation
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 160)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

def generate_embedding(model, face):
    # Convert face to tensor and preprocess
    face_tensor = preprocess(face).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = model(face_tensor).numpy()
    return embedding
