import torch
from PIL import Image as PILImage
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
stable_diffusion = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
stable_diffusion.to(device)

def upload_initial_image(image):
    pil_image = PILImage.open(image)
    return pil_image

def generate_image_from_text(text):
    image = stable_diffusion(text, num_inference_steps=20).images[0]
    return image