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
CORRECT_ANSWER = '52' # 정답설정
ANSWER_TOKEN = 'Answer: ' # 답을 나타내는 토큰
CODE_START_TOKEN = "# CODE START" # 코드 시작을 알리는 토큰
CODE_END_TOKEN = "# CODE END" # 코드 종료를 알리는 토큰
MAX_TOKENS = 4096 # 최대 토큰 수 설정
# ENCODER = tiktoken.encoding_for_model(ENGINE) # 모델에 맞는 토큰 인코딩 설정

original_image = "/home/jclee/workspace/src/gpt4o_api/medicin_pic/Grounding DINO Center.jpg"
# original_image = "/home/jclee/workspace/src/gpt4o_api/medicin_pic/GroundingDINO Center Bold.jpg" # bold가 생각보다 5,6을 헷갈림.
# original_image = "/home/jclee/workspace/src/gpt4o_api/medicin_pic/GroundingDINO bbox yellow.jpg"

# 로컬 이미지 파일을 Base64로 인코딩
with open(original_image, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

# Helper function to interact with GPT-4
def ask_gpt(prompt):
    
    response = client.chat.completions.create(
        model=ENGINE,
        messages=[
            {
                "role": "system", "content": "Let's think step by step. 생각은 영어로 하되, 대답은 한국어로만 해."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64, {base64_image}",
                        }
                    },
                ],
            },
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.5,
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

나 팔이 찢어졌어 아파. 이 상황에 적절한 약을 번호로 설명해줘. 이 때, 사진 내의 물체 별 중앙에 위치한 index 숫자 text를 참고해.
<예시 설명>
1. {'object_name': '물체이름1', 'info': '물체특징1', 'index_text': '물체 위에 그려진 숫자'}
2. {'object_name': '물체이름2', 'info': '물체특징2', 'index_text': '물체 위에 그려진 숫자'}
...
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



# PoT
"""
please use def find_yellow_text(image).
def find_yellow_text(image):
    import cv2
    import pytesseract
    import numpy as np

    # 이미지 파일 경로
    image_path = original_image

    # 이미지 읽기
    image = cv2.imread(image_path)

    # 이미지를 HSV 색 공간으로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 노란색 범위 정의 (HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # 노란색 영역 마스크 생성
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 원본 이미지와 마스크를 사용하여 노란색 부분만 추출
    yellow_text_region = cv2.bitwise_and(image, image, mask=yellow_mask)

    # 흑백 변환 (OCR 인식을 위해)
    gray = cv2.cvtColor(yellow_text_region, cv2.COLOR_BGR2GRAY)

    # 임계값 처리로 텍스트 영역 강조
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tesseract를 사용하여 이미지에서 텍스트 추출
    custom_config = r'--oem 3 --psm 6'  # 기본 설정
    extracted_text = pytesseract.image_to_string(thresh, config=custom_config, lang='kor')

    # 추출된 텍스트 출력
    print("추출된 텍스트:")
    print(extracted_text)

    # 결과 이미지와 마스크를 시각화하여 확인 (선택 사항)
    cv2.imshow('Original Image', image)
    cv2.imshow('Yellow Text Region', yellow_text_region)
    cv2.imshow('Thresholded', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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