import gradio as gr
from PIL import Image

def load_initial_image(image_path):
    try:
        initial_image = Image.open(image_path)
        return initial_image
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found.")
        return None

# 초기 이미지 파일 경로
initial_image_path = "initial_image.png"

# 초기 이미지 로드
initial_image_loaded = load_initial_image(initial_image_path)

# 이미지를 입력받아 그대로 출력하는 함수
def display_image(image):
    return image

# Gradio 인터페이스 정의
iface = gr.Interface(
    fn=display_image,
    inputs="image",
    outputs="image",
    title="Image Viewer",
    description="Load and display images.",
    examples=[[initial_image_loaded]],  # 초기 이미지를 예시로 설정
)

# 웹 애플리케이션 실행
iface.launch()