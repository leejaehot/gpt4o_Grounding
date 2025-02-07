# main.py

from main_gd import cls_agnostic_detection
from main_gpt4o import run_gpt4o_pipeline
import time
import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    # .env 에 기재된 OPENAI_API_KEY가 os.environ에 환경변수로 저장.
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
    if OPENAI_API_KEY is None:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env.")
    
    s_t = time.time()
    
    # 1) GroundingDINO 모델을 이용한 객체 검출 및 bbox index 마킹
    original_image_path = "/home/jclee/workspace/src/gpt4o_GD/input_img/medicine_image.jpeg"
    
    cls_agnostic_detection(
        input_img_path = original_image_path,
        text_query = "object.", # 기본값으로 "object."를 class-agnostic detector 역할로 사용.
        output_img_path = "/home/jclee/workspace/src/gpt4o_GD/output_img/gd_result.jpg",
        mapping_file_path = "/home/jclee/workspace/src/gpt4o_GD/mapping_json/proposal_mapping.json",
    )
    
    # 2) gpt4o_client를 통해 GPT-4o API로 질의
    openai_api_key = OPENAI_API_KEY  # 자신의 키로 대체
    
    user_prompt = ["""
       나 기침을 콜록콜록해. 
    """]
    
    run_gpt4o_pipeline(
        openai_api_key = openai_api_key,
        original_image_path = original_image_path,
        indexing_image_path = "/home/jclee/workspace/src/gpt4o_GD/output_img/gd_result.jpg",
        mapping_file_path = "/home/jclee/workspace/src/gpt4o_GD/mapping_json/proposal_mapping.json",
        user_prompt = user_prompt,
    )
    
    e_t = time.time()
    latency = e_t - s_t
    print(f"\n Total Latency: {latency:.4f} seconds")
    
if __name__ == "__main__":
    main()
