# detection/visualization.py

from PIL import ImageDraw, ImageFont
from tqdm import tqdm

def draw_boxes_with_index(image, results, font_path: str, font_size: int):
    """
    검출된 결과에 대해 bounding box와 중앙에 인덱스 번호를 그림.
    인덱스 번호는 검정 배경에 노란색 글씨로 표시.
    
    Args:
        image (PIL.Image): 원본 이미지.
        results (dict): 검출 결과 (post_process 결과).
        font_path (str): 텍스트 표시용 폰트 파일 경로.
        font_size (int): 폰트 크기.
        
    Returns:
        이미지 (PIL.Image): 박스와 인덱스가 그려진 이미지.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    boxes = results[0]["boxes"]
    
    # tqdm을 사용해 진행 상황 표시 (총 박스 개수)
    for idx, (box, label, score) in enumerate(
        tqdm(zip(boxes, results[0]["labels"], results[0]["scores"]), total=len(boxes)),
        start=1
    ):
        # 박스 좌표
        x0, y0, x1, y1 = box.tolist()
        draw.rectangle([x0, y0, x1, y1], outline="yellow", width=20)

        # bbox 중앙 좌표 계산
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2

        # 인덱스 번호 문자열
        index_text = str(idx)
        # 텍스트의 크기 측정을 위해 textbbox 사용 (좌표: (left, top, right, bottom))
        text_bbox = draw.textbbox((0, 0), index_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # 여백(margin) 포함하여 배경 사각형 좌표 계산
        margin = 5
        rect_x0 = center_x - text_width / 2 - margin
        rect_y0 = center_y - text_height / 2 - margin
        rect_x1 = center_x + text_width / 2 + margin
        rect_y1 = center_y + text_height / 2 + margin*2

        # 검정 배경 사각형 그리기
        draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill="black")
        # 중앙에 노란색 인덱스 번호 텍스트 그리기
        draw.text((center_x - text_width / 2, center_y - text_height / 2), index_text, fill="yellow", font=font)
        
    return image
