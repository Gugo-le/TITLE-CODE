import numpy as np
from PIL import Image

def mini_img(img_path, resize_shape=(10, 10)):
    img = Image.open(img_path)
    img = img.resize(resize_shape)
    return img

def cosine_similarity(img1, img2):
    array1 = np.array(img1)
    array2 = np.array(img2)
    assert array1.shape == array2.shape
    
    h, w, c = array1.shape
    len_vec = h * w * c
    vector_1 = array1.reshape(len_vec,) / 255.0
    vector_2 = array2.reshape(len_vec,) / 255.0

    cosine_similarity = np.dot(vector_1, vector_2) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2))
    return cosine_similarity

img1_path = "./assets/imgs/chat.jpg"
img2_path = "./assets/imgs/gpt.jpg"

img1 = mini_img(img1_path)
img2 = mini_img(img2_path)

score = cosine_similarity(img1, img2)
print("두 이미지의 유사도:", score)