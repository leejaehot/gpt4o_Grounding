# image_utils.py

import base64

def image_file_to_base64(image_path: str) -> str:
    """
    로컬 이미지 파일을 읽어 base64 인코딩 문자열로 변환합니다.
    
    Args:
        image_path (str): 이미지 파일 경로.
        
    Returns:
        str: base64 인코딩된 문자열.
    """
    with open(image_path, "rb") as image_file:
        b64_str = base64.b64encode(image_file.read()).decode('utf-8')
    return b64_str
