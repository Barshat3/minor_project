import numpy as np

def cosine_similarity(embedding1, embedding2):
    # Compute cosine similarity between two embeddings
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