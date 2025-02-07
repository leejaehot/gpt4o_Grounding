# main_gd.py

import os
import torch
from PIL import Image
from model_utils import load_processor_and_model
from detection_utils import run_detection
from visualization import draw_boxes_with_index

from mapping import create_mapping_from_detections, save_mapping_json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="6"
os.environ["OMP_NUM_THREADS"]='1'

# GroundingDINO로 객체 검출 및 bbox-index 마킹을 수행하는 메인 파이프라인 함수.
def cls_agnostic_detection(
    input_img_path: str,
    text_query: str = "object.", # 기본값으로 "object."를 class-agnostic detector 역할로 사용.
    output_img_path: str = "/home/jclee/workspace/src/gpt4o_GD/output_img/gd_result.jpg",
    mapping_file_path: str = "/home/jclee/workspace/src/gpt4o_GD/mapping_json/proposal_mapping.json",
):
    """
    Args:
        input_img_path (str): 분석할 이미지 경로
        text_query (str): GroundingDINO에 넘길 텍스트 쿼리
        output_img_path (str): 결과 시각화 이미지를 저장할 경로
        mapping_file_path (str): bbox-index 매핑 정보를 저장할 JSON 경로
    """
        
    # 설정 값
    model_type = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_query = "object."  # 쿼리문은 소문자, 끝에 마침표가 있어야 함
    font_path = "Arial.ttf"  # 폰트 경로 (필요에 따라 경로 수정)
    font_size = 50
    
    # 1) 이미지 로드
    image = Image.open(input_img_path)
    
    # 2) 모델과 프로세서 로드
    processor, model = load_processor_and_model(model_type, device)
    
    # 3) 객체 검출 수행
    results = run_detection(image, text_query, processor, model, device)
    
    # 4) bbox-index mapping 후 json으로 proposal_mapping.json 저장.
    mapping = create_mapping_from_detections(results[0]) # results -> bbox-index mapping
    mapping_file_path = "/home/jclee/workspace/src/gpt4o_GD/mapping_json/proposal_mapping.json"
    save_mapping_json(mapping, mapping_file_path)
    
    # 5) 결과 시각화: 박스와 인덱스 번호 그리기
    image_with_boxes = draw_boxes_with_index(image, results, font_path, font_size)
    
    # 6) 결과 이미지 저장
    image_with_boxes.save(output_img_path)
    print(f"[GroundingDINO] 결과 이미지가 저장되었습니다.\n: {output_img_path}\n")
    print(f"[GroundingDINO] bbox-index 매핑 JSON 저장되었습니다.\n: {mapping_file_path}\n")

if __name__ == "__main__":
    input_img_path = "/home/jclee/workspace/src/gpt4o_GD/input_img/medicine_image.jpeg"
    run_detection_pipeline(input_img_path)