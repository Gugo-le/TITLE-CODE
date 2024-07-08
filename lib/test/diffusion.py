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

for step in range(max_steps + 1):
    if step in steps_to_save:
        noise_images.append((step, np.copy(noisy_image)))
    noisy_image = add_color_noise(noisy_image, noise_level)
    
plt.figure(figsize=(15, 8))
for i, (step, img) in enumerate(noise_images):
    plt.subplot(2, 4, i + 1)
    plt.imshow(img)
    plt.title(f'Step {step}')
    plt.axis('off')
plt.tight_layout()
plt.show()
    