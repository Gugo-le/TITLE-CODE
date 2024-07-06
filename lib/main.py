import torch
from PIL import Image as PILImage
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import gradio as gr

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
stable_diffusion = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
stable_diffusion.to(device)

def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)

def upload_initial_image(image):
    pil_image = PILImage.open(image)
    return pil_image

def generate_image_from_text(text):
    image = stable_diffusion(text, num_inference_steps=20).images[0]
    return image

def compare_images(initial_image, generated_image):
    
    initial_image_features = clip_processor(images=initial_image, return_tensors="pt").pixel_values
    generated_image_features = clip_processor(images=generated_image, return_tensors="pt").pixel_values

    with torch.no_grad():
        initial_image_features = clip_model.get_image_features(initial_image_features.to(device))
        generated_image_features = clip_model.get_image_features(generated_image_features.to(device))
    
    
    similarity = torch.nn.functional.cosine_similarity(initial_image_features, generated_image_features).item() * 100

    return similarity


def gradio_interface():
    def process(initial_image, text):
        
        uploaded_initial_image = upload_initial_image(initial_image)

        generated_image = generate_image_from_text(text)

        similarity = compare_images(uploaded_initial_image, generated_image)

        return f"업로드한 초기 이미지와 생성된 이미지의 유사도: {similarity:.2f}%", uploaded_initial_image, generated_image

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
        title="이미지 업로드 및 생성 비교",
        description="초기 이미지를 업로드하고 텍스트를 입력하여 추상적인 이미지를 생성하고, 업로드한 초기 이미지와 생성된 이미지의 유사도를 비교합니다."
    )

    return interface

if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()
