# detection/model_utils.py

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def load_processor_and_model(model_id: str, device: str = "cpu"):
    """
    주어진 모델 ID에 대해 processor와 model을 로드합니다.
    
    Args:
        model_id (str): 사용할 모델의 Hugging Face ID.
        device (str): "cuda" 또는 "cpu".
        
    Returns:
        processor, model
    """
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return processor, model
