from openai import OpenAI
import os
import base64
import re, ast
import cv2


from dotenv import load_dotenv
load_dotenv()
# .env 에 기재된 OPENAI_API_KEY가 os.environ에 환경변수로 저장.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env.")
openai_api_key = OPENAI_API_KEY

client = OpenAI(
  api_key=OPENAI_API_KEY,  # this is also the default, it can be omitted
)


""" 글로벌 변수 설정 """
ENGINE = 'gpt-4o'
MAX_TOKENS = 4096 # 최대 토큰 수 설정
# ENCODER = tiktoken.encoding_for_model(ENGINE) # 모델에 맞는 토큰 인코딩 설정


indexing_image = "/home/jclee/workspace/src/gpt4o_GD/output_img/gd_result.jpg"
original_image = "/home/jclee/workspace/src/gpt4o_GD/input_img/medicine_image.jpeg"

# 로컬 이미지 파일을 Base64로 인코딩
with open(indexing_image, "rb") as image_file:
    base64_image_idx = base64.b64encode(image_file.read()).decode('utf-8')

# 로컬 이미지 파일을 Base64로 인코딩
with open(original_image, "rb") as image_file:
    base64_image_og = base64.b64encode(image_file.read()).decode('utf-8')

# Helper function to interact with GPT-4
def ask_gpt(prompt):
    
    system_prompt = """
    Let's think step by step. 생각은 영어로 하되, 대답은 한국어로만 해.
    """
    
    instruction_prompt = """
    다음은 OVD모델을 통해 나온 전경의 모든 물체들에 대한 class-agnostic한 bbox와 index 정보야.
    original image size = a*b
    Detected n bounding boxes.

    
    'index_text' : 1,
    'index_text' : 2,
    'index_text' : 3,
    'index_text' : 4,
    'index_text' : 5,
    'index_text' : 6,
    'index_text' : 7,
    'index_text' : 8,
    'index_text' : 9,
    'index_text' : 10,
    'index_text' : 11,
    'index_text' : 12,
    'index_text' : 13, 

    
    
    ------------------------------------

    이제 {User의 모호한 명령어}가 제시될 거야. original image와 indexing image를 모두 고려해서 이 상황에 적절한 물체를 번호로 설명해줘. 이 때, 사진 내의 물체 별 중앙에 위치한 index 숫자 text를 참고해. 무조건 예시 설명의 형식으로 대답해줘.
    <예시 설명>
    1. {'object_name': '물체이름1', 'info': '물체특징1', 'index_text': '물체 위에 그려진 모든 숫자'}
    2. {'object_name': '물체이름2', 'info': '물체특징2', 'index_text': '물체 위에 그려진 모든 숫자'}
    ...
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
                            "url": f"data:image/png;base64, {base64_image_og}",
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64, {base64_image_idx}",
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

prompt_list = [
    "책상 위에는 다양한 물건들이 어지럽게 놓여 있습니다. 중앙에는 캠벨 토마토 수프와 스팸 통조림, 참치 통조림이 나란히 놓여 있고, 그 옆에는 체즈잇 과자 상자와 블랙앤데커 전동 드릴이 함께 있습니다. 책상 뒤쪽으로는 컴퓨터 모니터와 키보드가 보이며, 책상 위에는 흰색 마커도 놓여 있습니다. 주변에는 전기 드릴과 간식, 식료품이 혼재되어 있어 작업 공간과 식품 보관이 섞여 있는 모습입니다."
]



# Simulating the conversation
for i, prompt in enumerate(prompt_list):
    print(f"User Prompt {i+1}: {prompt}")
    response = ask_gpt(prompt)
    print(f"GPT Response {i+1}:\n{response}\n")
