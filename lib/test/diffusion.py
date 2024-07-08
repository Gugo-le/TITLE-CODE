import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드
image_path = 'assets/imgs/cat.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR 포맷을 사용하므로 RGB로 변환

# 컬러 노이즈 추가 함수
def add_color_noise(img, noise_level):
    noise = np.random.normal(0, noise_level, img.shape)
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img, noise

# 노이즈 제거 함수
def remove_color_noise(noisy_img, noise_history):
    recovered_img = np.copy(noisy_img)
    for noise in reversed(noise_history):
        recovered_img = recovered_img - noise
        recovered_img = np.clip(recovered_img, 0, 255).astype(np.uint8)
    return recovered_img

# 노이즈를 추가할 단계
steps_to_save = [0, 10, 20, 100, 200, 300, 400, 500]
max_steps = 500
noise_level = 20  # 노이즈 강도를 높여봅니다. (한 단계당 추가되는 노이즈)

# 초기 이미지
noisy_image = np.copy(image)
noise_history = []

# 각 단계별로 노이즈를 추가한 이미지 생성
for step in range(max_steps + 1):
    if step in steps_to_save:
        noise_history.append(np.random.normal(0, noise_level, image.shape))  # 노이즈 기록
        noisy_image, _ = add_color_noise(noisy_image, noise_level)

# 결과 출력 (노이즈 추가 과정)
plt.figure(figsize=(15, 8))
for i, step in enumerate(steps_to_save):
    plt.subplot(2, 4, i + 1)
    plt.imshow(noisy_image)
    plt.title(f'Step {step}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# 노이즈 제거 과정
recovered_image = remove_color_noise(noisy_image, noise_history)

# 결과 출력 (노이즈 제거 과정)
plt.figure(figsize=(8, 8))
plt.imshow(recovered_image)
plt.title('Recovered Image')
plt.axis('off')
plt.show()