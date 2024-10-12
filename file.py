import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import pygame
from model import CNN


model = CNN()
model.load_state_dict(torch.load('drowsiness_model.pth'))
model.eval()

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((145, 145)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for built-in webcam, or change to the appropriate camera index

# Initialize Pygame for audio
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("siren-alert-96052.mp3")  # Replace with your alert sound file

# Variables for drowsiness detection
drowsy_threshold = 0.5
drowsy_duration_threshold = 2.0  # seconds
drowsy_start_time = None
alert_cooldown = 10.0  # seconds
last_alert_time = 0

def process_frame(frame):
    # Convert the frame to RGB (OpenCV uses BGR by default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Apply transformations
    input_tensor = transform(pil_image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        
        # Check the output shape
        print("Model output shape:", output.shape)  # Debugging line
        
        probabilities = torch.sigmoid(output).detach().numpy()  # Get probabilities for all classes
        
        # Check the probabilities shape
        print("Probabilities shape:", probabilities.shape)  # Debugging line
    
    return probabilities  # Return the array of probabilities

def alert_driver():
    print("ALERT: Driver drowsiness detected!")
    alert_sound.play()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        probabilities = process_frame(frame)

        # Correctly access the drowsiness score
        if probabilities.size == 5:
            drowsiness_score = probabilities[0][1]  # Access the first row (sample) and the 'drowsy' class
            
            current_time = time.time()
            
            if drowsiness_score > drowsy_threshold:
                if drowsy_start_time is None:
                    drowsy_start_time = current_time
                elif current_time - drowsy_start_time > drowsy_duration_threshold:
                    if current_time - last_alert_time > alert_cooldown:
                        alert_driver()
                        last_alert_time = current_time
            else:
                drowsy_start_time = None
            
            # Display drowsiness score on frame
            cv2.putText(frame, f'Drowsiness: {drowsiness_score:.2f}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if drowsiness_score > drowsy_threshold else (0, 255, 0), 2)
        else:
            print("Unexpected probabilities shape:", probabilities.shape)  # Handle unexpected shape

        # Display the frame
        import matplotlib.pyplot as plt

# Replace cv2.imshow with the following
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Hide the axis
        plt.show(block=False)  # Non-blocking show
        plt.pause(0.001)  # Pause to allow the image to display

        # Break the loop if 'q' is pressed
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

finally:
    # Release the camera and close all windows
    cap.release()