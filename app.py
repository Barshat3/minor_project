import cv2
import pickle
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
import torch

# Load the pre-trained FaceNet model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=20)

# Load saved embeddings
with open("embeddings.pickle", "rb") as f:
    stored_embeddings = pickle.load(f)

# Preprocessing transformation
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 160)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Function to compute cosine similarity
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

# Function to recognize a face
def recognize_face(embedding, stored_embeddings, threshold=0.6):
    best_match = None
    best_similarity = -1
    for name, embeddings_list in stored_embeddings.items():
        for stored_embedding in embeddings_list:
            similarity = cosine_similarity(embedding, stored_embedding)
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match = name
    return best_match if best_match else "Unknown"

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]  # Crop the face

            # Preprocess the face
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_tensor = preprocess(face_rgb).unsqueeze(0)  # Add batch dimension

            # Generate embedding for the detected face
            with torch.no_grad():
                embedding = facenet_model(face_tensor).squeeze().numpy()

            # Recognize the face
            name = recognize_face(embedding, stored_embeddings)

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Face Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
