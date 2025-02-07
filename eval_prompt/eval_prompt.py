from openai import OpenAI
import os
import base64
import re, ast
import cv2
import json
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()
# .env 에 기재된 OPENAI_API_KEY가 os.environ에 환경변수로 저장.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env.")

client = OpenAI(
  api_key=OPENAI_API_KEY,  # this is also the default, it can be omitted
)


""" 글로벌 변수 설정 """
ENGINE = 'gpt-4o'
MAX_TOKENS = 4096 # 최대 토큰 수 설정
# ENCODER = tiktoken.encoding_for_model(ENGINE) # 모델에 맞는 토큰 인코딩 설정


center_indexing_image = "/Users/leejaehot/leejaehot_main/sejong🏫/4-2/패턴인식(재)/a100_backup/src/gpt4o_api/medicin_pic/Grounding DINO Center.jpg"
center_indexing_bold_image = "/Users/leejaehot/leejaehot_main/sejong🏫/4-2/패턴인식(재)/a100_backup/src/gpt4o_api/medicin_pic/GroundingDINO Center Bold.jpg"
jwasang_indexing_image = "/Users/leejaehot/leejaehot_main/sejong🏫/4-2/패턴인식(재)/a100_backup/src/gpt4o_api/medicin_pic/jwasang.jpg"
jwaha_indexing_image = "/Users/leejaehot/leejaehot_main/sejong🏫/4-2/패턴인식(재)/a100_backup/src/gpt4o_api/medicin_pic/jwaha.jpg"
woosang_indexing_image = "/Users/leejaehot/leejaehot_main/sejong🏫/4-2/패턴인식(재)/a100_backup/src/gpt4o_api/medicin_pic/woosang.jpg"
wooha_indexing_image = "/Users/leejaehot/leejaehot_main/sejong🏫/4-2/패턴인식(재)/a100_backup/src/gpt4o_api/medicin_pic/wooha.jpg"
yellow_bbox_indexing_image = "/Users/leejaehot/leejaehot_main/sejong🏫/4-2/패턴인식(재)/a100_backup/src/gpt4o_api/medicin_pic/GroundingDINO bbox yellow.jpg"


indexing_image = center_indexing_image
original_image = "/Users/leejaehot/leejaehot_main/sejong🏫/4-2/패턴인식(재)/a100_backup/src/gpt4o_api/medicine_image.jpeg"

# 로컬 이미지 파일을 Base64로 인코딩
with open(indexing_image, "rb") as image_file:
    base64_image_idx = base64.b64encode(image_file.read()).decode('utf-8')

# 로컬 이미지 파일을 Base64로 인코딩
with open(original_image, "rb") as image_file:
    base64_image_og = base64.b64encode(image_file.read()).decode('utf-8')

# Helper function to interact with GPT-4
def ask_gpt(prompt):
    
    system_prompt = """
    Let's think step by step. 생각은 영어로 하되, 대답은 한국어로만 해. 넌 지금부터 약사야.
    """
    
    instruction_prompt = """
    다음은 OVD모델을 통해 나온 전경의 모든 물체들에 대한 bbox와 index 정보야.
    original image size = 4032*3024
    Detected 16 bounding boxes.

    'index_text' : 1, 'bbox' : {x_min=2106, y_min=303, x_max=2992, y_max=880}
    'index_text' : 2, 'bbox' : {x_min=1710, y_min=155, x_max=1983, y_max=890}
    'index_text' : 3, 'bbox' : {x_min=3212, y_min=1076, x_max=3565, y_max=1966}
    'index_text' : 4, 'bbox' : {x_min=2529, y_min=1950, x_max=3949, y_max=2995}
    'index_text' : 5, 'bbox' : {x_min=1801, y_min=1820, x_max=2543, y_max=2932}
    'index_text' : 6, 'bbox' : {x_min=1118, y_min=1806, x_max=1816, y_max=2988}
    'index_text' : 7, 'bbox' : {x_min=1003, y_min=582, x_max=1607, y_max=753}
    'index_text' : 8, 'bbox' : {x_min=183, y_min=1847, x_max=1122, y_max=2443}
    'index_text' : 9, 'bbox' : {x_min=2570, y_min=929, x_max=3181, y_max=1951}
    'index_text' : 10, 'bbox' : {x_min=1216, y_min=1321, x_max=2025, y_max=1799}
    'index_text' : 11, 'bbox' : {x_min=287, y_min=2391, x_max=1118, y_max=2934}
    'index_text' : 12, 'bbox' : {x_min=217, y_min=880, x_max=1134, y_max=1384}
    'index_text' : 13, 'bbox' : {x_min=234, y_min=1322, x_max=1210, y_max=1888}
    'index_text' : 14, 'bbox' : {x_min=2042, y_min=942, x_max=2567, y_max=1805}
    'index_text' : 15, 'bbox' : {x_min=274, y_min=206, x_max=821, y_max=822}
    'index_text' : 16, 'bbox' : {x_min=1172, y_min=890, x_max=2001, y_max=1364}
    
    ------------------------------------

    이제 {User의 모호한 명령어}가 제시될 거야. original image와 indexing image를 모두 고려해서 이 상황에 적절한 약을 번호로 설명해줘. 이 때, 사진 내의 물체 별 중앙에 위치한 index 숫자 text를 참고해. 무조건 예시 설명의 형식으로 대답해줘. don't output the json script.
    <예시 설명>
    answer : {'object_name': '물체이름1', 'info': '물체특징1', 'index_text': '물체 위에 그려진 숫자'}
    
    if, 적절한 게 없으면 answer : {'object_name': "None", 'info': '', 'index_text': ''}
    """
    
    response = client.chat.completions.create(
        model=ENGINE,
        messages=[
            {
                "role": "system", "content": system_prompt
            },
            {
                "role": "user", "content": instruction_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64, {base64_image_og}",
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64, {base64_image_idx}",
                        }
                    },
                ],
            },
        ],
        max_tokens=MAX_TOKENS,
        temperature=0,
    )
    
    # breakpoint()
    return response.choices[0].message.content

# JSON 파일 로드 함수
def load_json_file(file_path):
    """JSON 파일에서 데이터를 불러옵니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []

# grounding_DINO_bbox.json 로드 함수
def load_grounding_dino_bbox(file_path):
    """Grounding DINO BBox 데이터를 로드합니다."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error loading grounding DINO bbox JSON file: {e}")
        return {}

def parse_response(response):
    """
    GPT-4의 응답에서 object_name, info, index_text를 파싱합니다.
    response: GPT-4의 응답 문자열
    """
    try:
        # 'answer :' 이후의 텍스트만 추출
        if "answer :" in response:
            response = response.split("answer :", 1)[1].strip()
        
        # 추출한 텍스트를 딕셔너리로 변환
        parsed_response = ast.literal_eval(response)
        return parsed_response
    except Exception as e:
        print(f"Error parsing response: {e}")
        return {}

# 모델 평가 함수
def evaluate_model(data, grounding_dino_data):
    """데이터셋을 사용하여 GPT-4 모델 평가를 수행합니다."""
    correct = 0
    total = len(data)
    # object_name -> index_text 매핑 생성
    object_to_index_map = {
        str(item["object_name"]): str(item["index_text"])
        for item in grounding_dino_data.get("detected_bounding_boxes", [])
    }
    
    for item in tqdm(data):
        query = item["query"]
        gt = item["gt"]
        expected_index = []
        for gt_name in gt:
            expected_index.append(object_to_index_map.get(gt_name, ""))  # gt[0]과 매핑된 index_text
        
        # GPT-4 호출
        try:
            response = ask_gpt(query)
            print(f"GPT Response for Index {item['index']}, Query '{query}':\n{response}")
        except Exception as e:
            print(f"Error querying GPT-4 for index {item['index']}: {e}")
            continue
        
        # response를 dictionary로 변환
        try:
            response_dict = parse_response(response)  # 안전하게 문자열을 Python dict로 변환
        except (ValueError, SyntaxError):
            print(f"Invalid response format for index {item['index']}: {response}")
            continue
        
        
        # object_name과 index_text 비교
        if (
            response_dict.get("object_name") in gt  # object_name이 gt와 같음
            and response_dict.get("index_text") in expected_index  # index_text가 사전정의된 index와 같음
        ):
            correct += 1
        print(f"gt : {gt}")
        print(f"correct/total = {correct}/{total}\n")
    
    # 정확도 계산
    accuracy = correct / total if total > 0 else 0
    return accuracy

# 메인 실행 코드
if __name__ == "__main__":
    # JSON 파일 경로
    eval_file_path = "/Users/leejaehot/leejaehot_main/sejong🏫/4-2/패턴인식(재)/a100_backup/src/gpt4o_api/eval_prompt/prompt_dataset_level4.json"
    grounding_dino_path = "/Users/leejaehot/leejaehot_main/sejong🏫/4-2/패턴인식(재)/a100_backup/src/gpt4o_api/eval_prompt/grounding_DINO_bbox.json"
    
    # 데이터 로드
    data = load_json_file(eval_file_path)
    grounding_dino_data = load_grounding_dino_bbox(grounding_dino_path)
    if not data or not grounding_dino_data:
        print("Failed to load data or grounding DINO bounding box data. Exiting.")
    else:
        # 모델 평가
        accuracy = evaluate_model(data, grounding_dino_data)
        print(f"Accuracy: {accuracy:.2f}")
        