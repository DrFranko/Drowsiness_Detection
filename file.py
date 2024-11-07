import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import pygame
import matplotlib.pyplot as plt
from model import CNN

model = CNN()
model.load_state_dict(torch.load('drowsiness_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((145, 145)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cap = cv2.VideoCapture(0) 

pygame.mixer.init()
alert_sound = pygame.mixer.Sound("siren-alert-96052.mp3")  

drowsy_threshold = 0.5
drowsy_duration_threshold = 2.0 
drowsy_start_time = None
alert_cooldown = 10.0  
last_alert_time = 0

start_time = time.time()
MAX_RUNTIME = 15  # seconds

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(rgb_frame)
    input_tensor = transform(pil_image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.sigmoid(output).detach().numpy()  
    
    return probabilities  

def alert_driver():
    print("ALERT: Driver drowsiness detected!")
    alert_sound.play()

try:
    while True:
        if time.time() - start_time > MAX_RUNTIME:
            print("Maximum runtime reached. Stopping detection.")
            break

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        probabilities = process_frame(frame)

        if probabilities.size == 5:
            drowsiness_score = probabilities[0][1] 
            
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
            
            cv2.putText(frame, f'Drowsiness: {drowsiness_score:.2f}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0, 0, 255) if drowsiness_score > drowsy_threshold else (0, 255, 0), 2)
        else:
            print("Unexpected probabilities shape:", probabilities.shape)

        plt.clf()  # Clear the previous figure
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off') 
        plt.draw()
        plt.pause(0.001) 

        if plt.waitforbuttonpress(timeout=0.001):
            break

finally:
    cap.release()
    plt.close()
    pygame.mixer.quit()

print("Drowsiness detection complete.")