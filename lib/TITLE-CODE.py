import torch
from PIL import Image as PILImage
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import gradio as gr

# CLIP 모델과 프로세서 로드
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Stable Diffusion 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
stable_diffusion = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
stable_diffusion.to(device)

# 안전 체크 함수 비활성화
def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)
stable_diffusion.safety_checker = dummy_safety_checker

# 초기 이미지를 업로드하는 함수
def upload_initial_image(image):
    pil_image = PILImage.open(image)
    return pil_image

# 텍스트 설명을 기반으로 이미지를 생성하는 함수
def generate_image_from_text(text):
    image = stable_diffusion(text, num_inference_steps=20).images[0]
    return image

# 두 이미지를 비교하여 유사도를 계산하는 함수
def compare_images(initial_image, generated_image):
    # 이미지 특징 추출
    initial_image_features = clip_processor(images=initial_image, return_tensors="pt").pixel_values
    generated_image_features = clip_processor(images=generated_image, return_tensors="pt").pixel_values

    with torch.no_grad():
        initial_image_features = clip_model.get_image_features(initial_image_features.to(device))
        generated_image_features = clip_model.get_image_features(generated_image_features.to(device))
    
    # 코사인 유사도 계산
    similarity = torch.nn.functional.cosine_similarity(initial_image_features, generated_image_features).item() * 100

    return similarity


# Gradio 인터페이스 설정
def gradio_interface():
    def process(initial_image, text):
        
        # 초기 이미지 업로드
        uploaded_initial_image = upload_initial_image(initial_image)

        # 텍스트를 기반으로 이미지 생성
        generated_image = generate_image_from_text(text)

        # 이미지 유사도 비교
        similarity = compare_images(uploaded_initial_image, generated_image)

        return f"업로드한 초기 이미지와 생성된 이미지의 유사도: {similarity:.2f}%", generated_image

    inputs = [
        gr.Image(label="초기 이미지 업로드", type="filepath"),
        gr.Textbox(label="텍스트 입력", placeholder="Enter a description to generate an image")
    ]

    outputs = [
        gr.Textbox(label="유사도 결과"),
        gr.Image(label="생성된 이미지")
    ]

    interface = gr.Interface(
        fn=process,
        inputs=inputs,
        outputs=outputs,
        title="TITLE-CODE",
        description="100년 후 미래에는 질문 능력이 중요하지 않을까?"
    )

    return interface

if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()