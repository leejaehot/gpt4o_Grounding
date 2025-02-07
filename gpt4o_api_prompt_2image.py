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
  api_key=openai_api_key,  # this is also the default, it can be omitted
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
    'index_text' : 14, 
    'index_text' : 15, 
    'index_text' : 16,
    
    
    ------------------------------------

    이제 {User의 모호한 명령어}가 제시될 거야. original image와 indexing image를 모두 고려해서 이 상황에 적절한 약을 번호로 설명해줘. 이 때, 사진 내의 물체 별 중앙에 위치한 index 숫자 text를 참고해. 무조건 예시 설명의 형식으로 대답해줘.
    <예시 설명>
    1. {'object_name': '물체이름1', 'info': '물체특징1', 'index_text': '물체 위에 그려진 숫자'}
    2. {'object_name': '물체이름2', 'info': '물체특징2', 'index_text': '물체 위에 그려진 숫자'}
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


#Initial conversation
prompt_list = [
    # "이미지의 좌상단부터, 보이는 약마다 약 위에 순서대로 번호를 매겨서 우하단의 약까지 번호가 다 매겨지게 해봐.",
    "이미지 내에 보이는 모든 물체를 말해줘.",
    # "스트렙실 놓쳤어. 다시.",
    "나 팔이 찢겨서 피가 철철 나. 이 상황에 필요한 물체를 번호로 불러줘. \
        다음과 같은 예시로 설명해. \
        1. {'object': '물체이름1', 'info': '물체특징1', 'position coords': 이미지 내에서 위치하는 point1([x_norm,y_norm])} \
        2. {'object': '물체이름2', 'info': '물체특징2', 'position coords': 이미지 내에서 위치하는 point2([x_norm,y_norm])} \
        3. {'object': '물체이름3', 'info': '물체특징3', 'position coords': 이미지 내에서 위치하는 point3([x_norm,y_norm])} \
        (여기엔 적절한 행동 순서를 설명해줘).",
    # "이미지에서 좌상단부터 우상단까지 번호를 매겼던 것을 떠올려. 이젠 너가 제시한 물체에 해당하는 번호들을 ex) {1,2,5 ..}, 이런 형식으로 내뱉어줘.",
]

prompt_list = [
    "넌 약사야.",
    "나 지금 긁혀서 피나. 이 상황에 적절한 약을 번호로 설명해줘.",
    "Let's think step by step. 필요한 물체와 물체를 가지고 어떻게 행동할지, 다음과 같은 예시로 설명해. \
        <예시 설명> \
        1. {'object': '물체이름1', 'info': '물체특징1'} \
        2. {'object': '물체이름2', 'info': '물체특징2'} \
        3. {'object': '물체이름3', 'info': '물체특징3'} \
        ...",
    #"너가 언급해 준 객체들을 {'Apple', 'Banana', 'Carrot' ...} 의 형태로 표현해줘."
]

prompt_list = [
    "넌 약사야.",
    "나 머리가 깨질 것 같아. 이 상황에 적절한 약을 번호로 설명해줘.",
    "Let's think step by step. 필요한 물체와 물체를 가지고 어떻게 행동할지, 다음과 같은 예시로 설명해. \
        <예시 설명> \
        1. {'object': '물체이름1', 'info': '물체특징1'} \
        2. {'object': '물체이름2', 'info': '물체특징2'} \
        3. {'object': '물체이름3', 'info': '물체특징3'} \
        ...",
    #"너가 언급해 준 객체들을 {'Apple', 'Banana', 'Carrot' ...} 의 형태로 표현해줘."
]

prompt_list = [
    "넌 약사야.",
    "나 목이 너무 아파.",
    "Let's think step by step. 필요한 물체와 물체를 가지고 어떻게 행동할지, 다음과 같은 예시로 설명해. \
        <예시 설명> \
        1. {'object': '물체이름1', 'info': '물체특징1'} \
        2. {'object': '물체이름2', 'info': '물체특징2'} \
        3. {'object': '물체이름3', 'info': '물체특징3'} \
        ...",
    #"너가 언급해 준 객체들을 {'Apple', 'Banana', 'Carrot' ...} 의 형태로 표현해줘."
]


prompt_list = [
    "넌 약사야.",
    "나 어지러워.",
    "Let's think step by step. 필요한 물체와 물체를 가지고 어떻게 행동할지, 다음과 같은 예시로 설명해. \
        <예시 설명> \
        1. {'object': '물체이름1', 'info': '물체특징1'} \
        2. {'object': '물체이름2', 'info': '물체특징2'} \
        3. {'object': '물체이름3', 'info': '물체특징3'} \
        ...",
    #"너가 언급해 준 객체들을 {'Apple', 'Banana', 'Carrot' ...} 의 형태로 표현해줘."
]

prompt_list = [
    "넌 약사야.",
    "어제 축구하고 왔는데, 허리가 너무 아프다..",
    "Let's think step by step. 필요한 물체와 물체를 가지고 어떻게 행동할지, 다음과 같은 예시로 설명해. \
        <예시 설명> \
        1. {'object': '물체이름1', 'info': '물체특징1'} \
        2. {'object': '물체이름2', 'info': '물체특징2'} \
        3. {'object': '물체이름3', 'info': '물체특징3'} \
        ...",
    #"너가 언급해 준 객체들을 {'Apple', 'Banana', 'Carrot' ...} 의 형태로 표현해줘."
]

prompt_list = [
    "머리가 지끈지끈해.",
    "Let's think step by step. 좀 전에 알려준 객체를, 다음과 같은 예시로 설명해. \
        <예시 설명> \
        1. {'object': '물체이름1', 'info': '물체특징1'} \
        2. {'object': '물체이름2', 'info': '물체특징2'} \
        ...",
    #"너가 언급해 준 객체들을 {'Apple', 'Banana', 'Carrot' ...} 의 형태로 표현해줘."
]

prompt_list = [
    "당신은 이미지 분석과 제품 분류에 능숙한 전문가입니다. 머리 아플 때 먹는 약 좀 줘.",
    "Let's think step by step. 이미지 내의 객체를, 다음과 같은 예시로 설명해. \
        <예시 설명> \
        1. {'object': '물체이름1', 'info': '물체특징1'} \
        2. {'object': '물체이름2', 'info': '물체특징2'} \
        ...",
    #"너가 언급해 준 객체들을 {'Apple', 'Banana', 'Carrot' ...} 의 형태로 표현해줘."
]

prompt_list = ["""General Structure for Zero-Shot Image Analysis Prompt:
Goal: Describe and analyze the products in the image comprehensively, focusing on names, types, ingredients, and uses.
Prompt Example: 'You are an expert in image analysis and product categorization. Analyze the image provided. Focus on the following tasks:
Identify Products: Extract the names of all visible items in the image, including brands and labels.
Categorize: Describe each item by type (e.g., medicine, first-aid, personal care).
List Ingredients or Components: Deduce or identify the active ingredients or key components based on visible text or common knowledge.
Describe Uses: Explain the likely purpose or application of each product (e.g., pain relief, wound care, or hydration).
Contextual Insights: Infer the setting or scenario related to the items, such as a first-aid kit or personal care package.*
Use a structured response format for clarity.'"""]




prompt_list = [
# """
# 이미지 내에서 보이는 약마다 번호를 매긴 걸 보고, 나에게 물체의 종류와 번호를 mapping해서 알려줘.
# """
# ,
"""
나 팔이 찢어졌어 아파.
"""


]

# 팔 찢어짐 prompt
"""
나 팔이 찢어졌어 아파. 이 상황에 적절한 약을 번호로 설명해줘. 이 때, 사진 내의 물체 별 좌상단에 위치한 노란색 배경의 까만 index 숫자 text를 참고해.
<예시 설명>
1. {'object_name': '물체이름1', 'info': '물체특징1', 'index_text': '물체 위에 그려진 숫자'}
2. {'object_name': '물체이름2', 'info': '물체특징2', 'index_text': '물체 위에 그려진 숫자'}
...
"""

# 머리 깨질 것 같아 prompt
"""
나 머리가 깨질 것 같아. 이 상황에 적절한 약을 번호로 설명해줘. 이 때, 사진 내의 물체 별 중앙에 위치한 index 숫자 text를 참고해.
<예시 설명>
1. {'object_name': '물체이름1', 'info': '물체특징1', 'index_text': '물체 위에 그려진 숫자'}
2. {'object_name': '물체이름2', 'info': '물체특징2', 'index_text': '물체 위에 그려진 숫자'}
...
"""

# 머리 지끈지끈 prompt
"""
나 머리가 지끈지끈해. 이 상황에 적절한 약을 번호로 설명해줘. 이 때, 사진 내의 물체 별 중앙에 위치한 index 숫자 text를 참고해.
<예시 설명>
1. {'object_name': '물체이름1', 'info': '물체특징1', 'index_text': '물체 위에 그려진 숫자'}
2. {'object_name': '물체이름2', 'info': '물체특징2', 'index_text': '물체 위에 그려진 숫자'}
...
"""

# 목이 너무 칼칼해 prompt
"""
나 목이 너무 칼칼해. 이 상황에 적절한 약을 번호로 설명해줘. 이 때, 사진 내의 물체 별 중앙에 위치한 index 숫자 text를 참고해.
<예시 설명>
1. {'object_name': '물체이름1', 'info': '물체특징1', 'index_text': '물체 위에 그려진 숫자'}
2. {'object_name': '물체이름2', 'info': '물체특징2', 'index_text': '물체 위에 그려진 숫자'}
...
"""

# 나 자꾸 기침이 나와 prompt
"""
나 자꾸 기침이 나와. 이 상황에 적절한 약을 번호로 설명해줘. 이 때, 사진 내의 물체 별 중앙에 위치한 index 숫자 text를 참고해.
<예시 설명>
1. {'object_name': '물체이름1', 'info': '물체특징1', 'index_text': '물체 위에 그려진 숫자'}
2. {'object_name': '물체이름2', 'info': '물체특징2', 'index_text': '물체 위에 그려진 숫자'}
...
"""

# 아세트아미노펜이 없는 두통약을 골라줘. prompt
"""
아세트아미노펜이 없는 두통약을 골라줘.. 이 상황에 적절한 약을 번호로 설명해줘. 이 때, 사진 내의 물체 별 중앙에 위치한 index 숫자 text를 참고해.
<예시 설명>
1. {'object_name': '물체이름1', 'info': '물체특징1', 'index_text': '물체 위에 그려진 숫자'}
2. {'object_name': '물체이름2', 'info': '물체특징2', 'index_text': '물체 위에 그려진 숫자'}
...
"""

# 아 나 축구하다가 허리 삐었어. prompt
"""
아 나 축구하다가 허리 삐었어. 이 상황에 적절한 약을 번호로 설명해줘. 이 때, 사진 내의 물체 별 중앙에 위치한 index 숫자 text를 참고해.
<예시 설명>
1. {'object_name': '물체이름1', 'info': '물체특징1', 'index_text': '물체 위에 그려진 숫자'}
2. {'object_name': '물체이름2', 'info': '물체특징2', 'index_text': '물체 위에 그려진 숫자'}
...
"""



# Simulating the conversation
for i, prompt in enumerate(prompt_list):
    print(f"User Prompt {i+1}: {prompt}")
    response = ask_gpt(prompt)
    print(f"GPT Response {i+1}:\n{response}\n")
    
    # if i == 1:
    #     matches = re.findall(r"\{.*?\}", response)
    #     objects_info = [ast.literal_eval(match) for match in matches]
    #     image = cv2.imread(original_image)
    #     image_height, image_width = image.shape[:2]
    #     bbox_size = 50
    #     bbox_color = (0, 255, 0) # green
    #     bbox_thickness = 2
        
    #     for obj in objects_info:
    #         # 중심 좌표 추출
    #         center_x = int(obj['position coords'][0] * image_width)
    #         center_y = int(obj['position coords'][1] * image_height)
            
    #         # Bounding Box 좌표 계산
    #         start_x = max(center_x - bbox_size // 2, 0)
    #         start_y = max(center_y - bbox_size // 2, 0)
    #         end_x = min(center_x + bbox_size // 2, image_width)
    #         end_y = min(center_y + bbox_size // 2, image_height)
            
    #         # Bounding Box 그리기
    #         cv2.rectangle(image, (start_x, start_y), (end_x, end_y), bbox_color, bbox_thickness)
            
    #         # 객체 이름 표시
    #         label = obj['object']
    #         cv2.putText(image, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1)

    #     # 결과 이미지 저장
    #     output_path = "/home/jclee/workspace/src/gpt4o_api/output_image.jpg"  # 저장 경로
    #     cv2.imwrite(output_path, image)

    #     print(f"Bounding Box가 포함된 이미지를 저장했습니다: {output_path}")