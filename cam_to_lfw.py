import os
import cv2
from PIL import Image

# Initialize Viola-Jones Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def capture_and_save_image(output_folder):
    """
    Capture an image from the webcam, detect and crop the face, then save it to a folder named after the user.
    Args:
        output_folder (str): Path to save the captured and processed image.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)

    print("Press 'c' to capture the image or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Viola-Jones
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the video feed
        cv2.imshow("Webcam - Press 'c' to capture", frame)

        # Wait for user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(faces) > 0:
            # Capture and process the first detected face (can modify to process all faces)
            (x, y, w, h) = faces[0]
            face = frame[y:y+h, x:x+w]  # Crop the detected face

            # Resize the face to 250x250 pixels (LFW standard)
            face_resized = cv2.resize(face, (250, 250))

            # Convert to RGB format
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # Ask the user for the name
            name = input("Enter the name of the person: ")

            # Create a new folder with the name inside the output folder
            person_folder = os.path.join(output_folder, name)
            os.makedirs(person_folder, exist_ok=True)

            # Save the captured face in the person's folder
            output_image_path = os.path.join(person_folder, f"{name}_face.jpg")
            face_img = Image.fromarray(face_rgb)
            face_img.save(output_image_path)
            print(f"Image saved to {output_image_path}")
            break
        elif key == ord('q'):
            print("Quitting without saving.")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

output_folder = "/home/barshat/Desktop/minor_project/lfw"  # Replace with your desired output folder path
capture_and_save_image(output_folder)
