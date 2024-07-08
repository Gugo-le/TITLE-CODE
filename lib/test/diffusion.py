import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'assets/imgs/cat.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def add_color_noise(img, noise_level):
    noise = np.random.normal(0, noise_level, img.shape)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

steps_to_save = [0, 10, 20, 100, 300, 500, 700, 1000]
max_steps = 1000
noise_level = 10  

noisy_image = np.copy(image)
noise_images = []