# mapping.py

import json
import re
import ast

def load_mapping_json(mapping_path: str) -> dict:
    """
    JSON 파일로부터 index별 proposal bbox mapping 정보를 읽어옵니다.
    
    Args:
        mapping_path (str): mapping JSON 파일 경로.
        
    Returns:
        dict: mapping 정보가 담긴 딕셔너리.
    """
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return mapping

def mapping_json_to_string(mapping: dict) -> str:
    """
    mapping 딕셔너리를 사람이 읽기 좋은 JSON 문자열로 변환.
    
    Args:
        mapping (dict): mapping 정보 딕셔너리.
        
    Returns:
        str: 포맷팅된 JSON 문자열.
    """
    return json.dumps(mapping, indent=4, ensure_ascii=False)

def save_mapping_json(mapping: dict, save_path: str) -> None:
    """
    mapping 정보를 JSON 파일로 저장.
    
    Args:
        mapping (dict): mapping 정보 딕셔너리.
        save_path (str): 저장할 JSON 파일 경로.
    """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)

def create_mapping_from_detections(detections: dict) -> dict:
    """
    검출 결과에서 인덱스별 proposal mapping 정보를 생성.
    검출 결과의 각 bbox를 인덱스와 매핑하여 딕셔너리 형태로 반환.
    
    Args:
        detections (dict): 검출 결과 딕셔너리 (예: results[0] from grounding dino)
        
    Returns:
        dict: 인덱스별로 bbox mapping 정보를 담은 딕셔너리.
              예시: {
                  "1": {"bbox": [x0, y0, x1, y1]},
                  "2": {"bbox": [x0, y0, x1, y1]},
                  ...
              }
    """
    mapping = {}
    boxes = detections.get("boxes", [])
    for idx, box in enumerate(boxes, start=1):
        mapping[str(idx)] = {"bbox": box.tolist()}
    return mapping


def parse_gpt_response_to_dict(gpt_response: str):
    """
    주어진 gpt_response 문자열이 
    예) '```json\\n[ ... ]\\n```'
    형태로 들어올 때,
    - 백틱('```json', '```')을 제거하고
    - ast.literal_eval로 파이썬 객체(리스트/딕셔너리)로 변환해서 반환한다.
    """
    # 1) 문자열 앞뒤의 ```json / ``` 제거
    #    정규표현식으로 맨 앞의 ```json (및 뒤따르는 공백/개행),
    #    맨 뒤의 ``` 를 제거한다.
    text = re.sub(r'^```json\s*', '', gpt_response.strip())
    text = re.sub(r'```$', '', text.strip())
    
    # 2) ast.literal_eval로 파이썬 객체로 변환
    #    (JSON 형태지만, literal_eval도 정상 파싱 가능)
    try:
        data = ast.literal_eval(text)
        return data
    except Exception as e:
        print("[ERROR] ast.literal_eval 실패:", e)
        return None


def merge_gpt_response_with_mapping(gpt_response, mapping_json_path):
    """
    gpt_response: GPT에서 나온 응답(리스트 형태)
        예) [
            {'object_name': '투스쿨', 'info': '기침 완화', 'index_text': '4'},
            ...
        ]
    mapping_json_path: index -> bbox 매핑 정보를 담고 있는 JSON 파일 경로
    """
    # mapping_json 로드
    mapping_data = load_mapping_json(mapping_json_path)

    merged_results = []
    
    gpt_response = parse_gpt_response_to_dict(gpt_response)
    
    for item in gpt_response:
        # 'index_text' 값이 예: '4'
        index_str = item.get('index_text', None)
        
        # 매핑 JSON 안에 index_str이 존재하면 bbox 정보를 추가
        if index_str is not None and index_str in mapping_data:
            item['bbox'] = mapping_data[index_str]['bbox']
        else:
            item['bbox'] = None  # 매핑되지 않는다면 None 처리(또는 스킵)

        merged_results.append(item)
    
    merged_results = json.dumps(merged_results, ensure_ascii=False, indent=4)
    return merged_results