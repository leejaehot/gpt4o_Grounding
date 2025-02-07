# detection/detection_utils.py

import torch

def run_detection(image, text_query: str, processor, model, device: str,
                  box_threshold: float = 0.25, text_threshold: float = 0.25):
    """
    이미지와 텍스트 쿼리를 받아 객체 검출을 수행합니다.
    
    Args:
        image (PIL.Image): 입력 이미지.
        text_query (str): 텍스트 쿼리 (ex: "object.").
        processor: Hugging Face processor.
        model: Hugging Face 모델.
        device (str): "cuda" 또는 "cpu".
        box_threshold (float): 박스 임계값.
        text_threshold (float): 텍스트 임계값.
        
    Returns:
        결과 dictionary (post_process 결과)
    """
    inputs = processor(images=image, text=text_query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # target_sizes는 (height, width) 순서로 입력
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
    )
    return results
