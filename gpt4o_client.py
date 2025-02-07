# gpt4o_client.py

from openai import OpenAI

class GPT4o_Client:
    def __init__(self, api_key: str, engine: str = "gpt-4o", max_tokens: int = 4096, temperature: float = 0):
        """
        GPT-4o API 클라이언트를 초기화.
        
        Args:
            api_key (str): OpenAI API 키.
            engine (str): 사용할 모델 엔진 (기본값: "gpt-4o").
            max_tokens (int): 최대 토큰 수.
            temperature (float): 샘플링 온도.
        """
        self.api_key = api_key
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key)
    
    def ask(self, prompt: str, original_image_b64: str, indexing_image_b64: str, mapping_json_str: str) -> str:
        """
        GPT-4o API에 메시지를 보내고 응답을 반환합니다.
        
        Args:
            prompt (str): 사용자의 모호한 명령어.
            original_image_b64 (str): 원본 이미지의 base64 인코딩 문자열.
            indexing_image_b64 (str): index marked 이미지의 base64 인코딩 문자열.
            mapping_json_str (str): index별 proposal bbox mapping 정보를 담은 JSON 문자열.
            
        Returns:
            GPT-4O API의 응답 문자열.
        """
        system_prompt = (
            "Let's think step by step. 생각은 영어로 하되, 대답은 한국어로만 해."
        )
        
        instruction_prompt = f"""
다음은 OVD모델을 통해 나온 전경의 모든 물체들에 대한 class-agnostic한 bbox와 index 정보야.
original image size = a*b
Detected n bounding boxes.

{mapping_json_str}

------------------------------------

이제 {{User의 모호한 명령어}}가 제시될 거야. original image와 indexing image를 모두 고려해서 이 상황에 적절한 물체를 번호로 설명해줘. 
이 때, 사진 내의 물체 별 중앙에 위치한 index 숫자 text를 참고해.

When responding with your final answer, please output only valid JSON with the following structure:
[
    {{
        "object_name": "물체이름1",
        "info": "물체특징1",
        "index_text": "물체 위에 그려진 모든 숫자"
    }},
    {{
        "object_name": "물체이름2",
        "info": "물체특징2",
        "index_text": "물체 위에 그려진 모든 숫자"
    }}
]
No additional text outside the JSON.
"""
        # 메시지 구성: system, instruction, 그리고 사용자 입력(텍스트와 두 이미지)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{original_image_b64}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{indexing_image_b64}"
                    }
                },
            ]},
        ]
        
        response = self.client.chat.completions.create(
            model=self.engine,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content
