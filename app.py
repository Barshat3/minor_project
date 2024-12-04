import cv2
import pickle
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import torch
from collections import defaultdict, Counter
from recognition import cosine_similarity, recognize_face

# Load the pre-trained FaceNet model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Load the Viola-Jones classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load saved embeddings
with open("embeddings.pickle", "rb") as f:
    stored_embeddings = pickle.load(f)

# Preprocessing transformation for face images
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 160)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])


# Buffer to store recent predictions for each face
face_prediction_buffer = defaultdict(list)
buffer_size = 5  # Number of frames for consensus

# Helper function to find the closest face from the previous frame
def find_closest_face(new_face, existing_faces, threshold=50):
    """Find the closest existing face based on bounding box proximity."""
    for face_id, (x_prev, y_prev, w_prev, h_prev) in existing_faces.items():
        x, y, w, h = new_face
        distance = np.sqrt((x - x_prev)**2 + (y - y_prev)**2)
        if distance < threshold:
            return face_id
    return None

# Open the webcam
cap = cv2.VideoCapture(0)
existing_faces = {}  # Store face locations from the previous frame
face_id_counter = 0  # Unique ID for each detected face

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for Viola-Jones
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Viola-Jones
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    current_faces = {}

    for (x, y, w, h) in faces:
        # Find the closest existing face or assign a new ID
        face_id = find_closest_face((x, y, w, h), existing_faces)
        if face_id is None:
            face_id = face_id_counter
            face_id_counter += 1

        current_faces[face_id] = (x, y, w, h)

        # Crop the detected face
        face = frame[y:y+h, x:x+w]

        # Preprocess the face
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_tensor = preprocess(face_rgb).unsqueeze(0)  # Add batch dimension

        # Generate embedding for the detected face
        with torch.no_grad():
            embedding = facenet_model(face_tensor).squeeze().numpy()

        # Recognize the face
        name = recognize_face(embedding, stored_embeddings)

        # Add the name to the prediction buffer for this face
        face_prediction_buffer[face_id].append(name)

        # Limit the buffer size
        if len(face_prediction_buffer[face_id]) > buffer_size:
            face_prediction_buffer[face_id].pop(0)

        # Determine the most common prediction in the buffer
        if len(face_prediction_buffer[face_id]) == buffer_size:
            consensus_name = Counter(face_prediction_buffer[face_id]).most_common(1)[0][0]
        else:
            consensus_name = "Recognizing..."

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, consensus_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Update existing_faces with current_faces
    existing_faces = current_faces

    # Display the video feed
    cv2.imshow("Face Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
