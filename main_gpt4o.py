# main_gpt4o.py

from gpt4o_client import GPT4o_Client
from image_utils import image_file_to_base64
from mapping import load_mapping_json, mapping_json_to_string, merge_gpt_response_with_mapping
import os
from dotenv import load_dotenv

load_dotenv()
# .env 에 기재된 OPENAI_API_KEY가 os.environ에 환경변수로 저장.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

# GPT-4O API 질의를 수행하는 메인 파이프라인 함수.
def run_gpt4o_pipeline(
    openai_api_key: str,
    original_image_path: str,
    indexing_image_path: str = "/home/jclee/workspace/src/gpt4o_GD/output_img/gd_result.jpg",
    mapping_file_path: str = "/home/jclee/workspace/src/gpt4o_GD/mapping_json/proposal_mapping.json",
    user_prompt: list = [""" 나 머리가 지끈지끈 해. """],
):
    """
    Args:
        openai_api_key (str): OpenAI API 키
        original_image_path (str): 원본 이미지 경로
        indexing_image_path (str): bbox-index가 표시된 결과 이미지 경로
        mapping_file_path (str): bbox-index 매핑 JSON 경로
        user_prompt (str): GPT에게 전달할 사용자 명령어(질의)
    """
    
    # 이미지 파일을 base64 문자열로 변환
    original_image_b64 = image_file_to_base64(original_image_path)
    indexing_image_b64 = image_file_to_base64(indexing_image_path)
    
    # proposal mapping JSON 파일 로드 및 문자열 변환
    mapping = load_mapping_json(mapping_file_path)
    mapping_json_str = mapping_json_to_string(mapping)
    
    # GPT-4O 클라이언트 인스턴스 생성
    client = GPT4o_Client(api_key=openai_api_key)
    
# 책상 위에는 다양한 물건들이 어지럽게 놓여 있습니다. 중앙에는 캠벨 토마토 수프와 스팸 통조림, 참치 통조림이 나란히 놓여 있고, 그 옆에는 체즈잇 과자 상자와 블랙앤데커 전동 드릴이 함께 있습니다. 책상 뒤쪽으로는 컴퓨터 모니터와 키보드가 보이며, 책상 위에는 흰색 마커도 놓여 있습니다. 주변에는 전기 드릴과 간식, 식료품이 혼재되어 있어 작업 공간과 식품 보관이 섞여 있는 모습입니다.
    
    # 각 prompt에 대해 GPT-4O API에 질의하고 결과 출력
    for idx, prompt in enumerate(user_prompt, start=1):
        print(f"\nUser Prompt {idx}: {prompt}")
        response = client.ask(prompt, original_image_b64, indexing_image_b64, mapping_json_str)
        
        response = merge_gpt_response_with_mapping(response, mapping_file_path)
        
        print(f"\nGPT Response {idx}:\n{response}")

if __name__ == "__main__":
    openai_api_key = OPENAI_API_KEY  # 자신의 키로 대체
    input_original_image_path = "/home/jclee/workspace/src/gpt4o_GD/input_img/medicine_image.jpeg"
    user_prompt = [""" 나 머리가 지끈지끈 해. """]
    
    run_gpt4o_pipeline(
        openai_api_key = openai_api_key,
        original_image_path = input_original_image_path,
        indexing_image_path = "/home/jclee/workspace/src/gpt4o_GD/output_img/gd_result.jpg",
        mapping_file_path = "/home/jclee/workspace/src/gpt4o_GD/mapping_json/proposal_mapping.json",
        user_prompt = user_prompt,
    )