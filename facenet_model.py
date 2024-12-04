from facenet_pytorch import InceptionResnetV1
import torch

# Load the pre-trained FaceNet model
def load_facenet_model():
    # Initialize the model pre-trained on VGGFace2
    model = InceptionResnetV1(pretrained='vggface2').eval()
    return model
