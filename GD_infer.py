import requests

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm

import cv2
import os, sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="6"
os.environ["OMP_NUM_THREADS"]='1'

model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image = Image.open("/home/jclee/workspace/src/gpt4o_GD/input_img/medicine_image.jpeg")

# Check for cats and remote controls
# VERY important: text queries need to be lowercased + end with a dot
text = "object."

inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.25,
    text_threshold=0.25,
    target_sizes=[image.size[::-1]]
)


# 결과 시각화
draw = ImageDraw.Draw(image)
font_size = 50
font = ImageFont.truetype("Arial.ttf", font_size)

idx = 0
for box, label, score in tqdm(zip(results[0]["boxes"], results[0]["labels"], results[0]["scores"])):
    idx += 1
    # 박스 좌표
    x0, y0, x1, y1 = box.tolist()
    draw.rectangle([x0, y0, x1, y1], outline="yellow", width=20)
    
    # bbox 중앙 좌표 계산
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2

    # 인덱스 번호 문자열
    index_text = str(idx)
    
    # 인덱스 텍스트의 크기 측정 (배경 사각형 크기 결정용)
    text_bbox = draw.textbbox((0,0), index_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]  
    
    # 약간의 여백을 포함한 배경 사각형 좌표 계산
    margin = 5
    rect_x0 = center_x - text_width / 2 - margin
    rect_y0 = center_y - text_height / 2 - margin
    rect_x1 = center_x + text_width / 2 + margin
    rect_y1 = center_y + text_height / 2 + margin

    # 검정색 배경 사각형 그리기
    draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill="black")
    
    # 중앙에 노란색 인덱스 번호 텍스트 그리기
    draw.text((center_x - text_width / 2, center_y - text_height / 2), index_text, fill="yellow", font=font)

# 결과 이미지 저장
output_img_name = "gd_result.jpg"
image.save("/home/jclee/workspace/src/gpt4o_GD/output_img/"+output_img_name)
print(f"결과 이미지가 {output_img_name}로 저장되었습니다.")

