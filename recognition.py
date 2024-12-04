import numpy as np

def cosine_similarity(embedding1, embedding2):
    # Compute cosine similarity between two embeddings
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def recognize_face(embedding, stored_embeddings, threshold=0.6):
    # Compare the face embedding to stored embeddings
    for name, stored_embedding in stored_embeddings.items():
        similarity = cosine_similarity(embedding, stored_embedding)
        if similarity > threshold:
            return name  # Return the name of the person
    return "Unknown"
