# 1,2,3,4,5,6,7 .. 이 이미지 내 물체별 주어진 경우
prompt_list = [
    "나 지금 긁혀서 피나. 이 상황에 적절한 약을 번호로 제시해줘."
    "Let's think step by step. 필요한 물체와 물체를 가지고 어떻게 행동할지, 다음과 같은 예시로 설명해. \
        <예시 설명> \
        1. {'object': '물체이름1', 'info': '물체특징1'} \
        2. {'object': '물체이름2', 'info': '물체특징2'} \
        3. {'object': '물체이름3', 'info': '물체특징3'} \
        ..."
    "너가 언급해 준 객체들을 {'Apple', 'Banana', 'Carrot' ...} 의 형태로 표현해줘."
]
prompt_list1 = [
    "나 머리가 깨질 것 같아."
    "Let's think step by step. 필요한 물체와 물체를 가지고 어떻게 행동할지, 다음과 같은 예시로 설명해. \
        <예시 설명> \
        1. {'object': '물체이름1', 'info': '물체특징1'} \
        2. {'object': '물체이름2', 'info': '물체특징2'} \
        3. {'object': '물체이름3', 'info': '물체특징3'} \
        ..."
    "너가 언급해 준 객체들을 {'Apple', 'Banana', 'Carrot' ...} 의 형태로 표현해줘."
]
prompt_list2 = [
    "나 목이 너무 아파."
    "Let's think step by step. 필요한 물체와 물체를 가지고 어떻게 행동할지, 다음과 같은 예시로 설명해. \
        <예시 설명> \
        1. {'object': '물체이름1', 'info': '물체특징1'} \
        2. {'object': '물체이름2', 'info': '물체특징2'} \
        3. {'object': '물체이름3', 'info': '물체특징3'} \
        ..."
    "너가 언급해 준 객체들을 {'Apple', 'Banana', 'Carrot' ...} 의 형태로 표현해줘."
]
prompt_list3 = [
    "나 어지러워."
    "Let's think step by step. 필요한 물체와 물체를 가지고 어떻게 행동할지, 다음과 같은 예시로 설명해. \
        <예시 설명> \
        1. {'object': '물체이름1', 'info': '물체특징1'} \
        2. {'object': '물체이름2', 'info': '물체특징2'} \
        3. {'object': '물체이름3', 'info': '물체특징3'} \
        ..."
    "너가 언급해 준 객체들을 {'Apple', 'Banana', 'Carrot' ...} 의 형태로 표현해줘."
]
prompt_list4 = [
    "어제 축구하고 왔는데, 허리가 너무 아프다.."
    "Let's think step by step. 필요한 물체와 물체를 가지고 어떻게 행동할지, 다음과 같은 예시로 설명해. \
        <예시 설명> \
        1. {'object': '물체이름1', 'info': '물체특징1'} \
        2. {'object': '물체이름2', 'info': '물체특징2'} \
        3. {'object': '물체이름3', 'info': '물체특징3'} \
        ..."
    "너가 언급해 준 객체들을 {'Apple', 'Banana', 'Carrot' ...} 의 형태로 표현해줘."
]

prompt_list4 = [
    "머리가 지끈지끈해."
    "Let's think step by step. 필요한 물체와 물체를 가지고 어떻게 행동할지, 다음과 같은 예시로 설명해. \
        <예시 설명> \
        1. {'object': '물체이름1', 'info': '물체특징1'} \
        2. {'object': '물체이름2', 'info': '물체특징2'} \
        3. {'object': '물체이름3', 'info': '물체특징3'} \
        ..."
    "너가 언급해 준 객체들을 {'Apple', 'Banana', 'Carrot' ...} 의 형태로 표현해줘."
]















# grounding 시도...
prompt_list_2 = [
    "이미지의 좌상단부터, 보이는 약마다 약 위에 순서대로 번호를 매겨서 우하단의 약까지 번호가 다 매겨지게 해봐.",
    # "스트렙실 놓쳤어. 다시.",
    "나 팔이 찢겨서 피가 철철 나. 이 상황에 필요한 약을 번호로 불러줘. \
        다음과 같은 예시로 설명해. \
        <예시 설명> \
        1. **물체1**: 행동 1. (사진 내의 위치 bbox좌표값) \
        2. **물체2**: 행동 2. (사진 내의 위치 bbox좌표값) \
        3. **물체3**: 행동 3. (사진 내의 위치 bbox좌표값) \
        4. **물체4**: 행동 4. (사진 내의 위치 bbox좌표값) \
        ...\
        (사진 내의 위치는 ex: [0.15, 0.25, 0.45, 0.35]의 [x_min_norm, y_min_norm, x_max_norm, y_max_norm] 형태로 표현.) \
        물체에 따른 논리적인 행동 순서 설명.",
    # "이미지의 좌상단부터 우하단까지, 보이는 약마다 약 위에 순서대로 번호를 매겼었잖아?\
    "상황에 필요했던 물체를 이미지에서 할당되었던 번호와 결부지어서 object = {1,2,5,...}의 dict 형식으로 output을 줘. 즉 grouding 해줘.",
    
    # "이미지에서 상황별로 필요할 수 있는 물품들을 번호와 결부지어 리스트로 나타내면 다음과 같습니다:\
    #     ```python\
    #     objects = {\
    #         '물집 보호': [10, 11],  # 메디터치, 밴드\
    #         '진통제': [2],  # 타이레놀\
    #         '소독': [5, 9],  # 과산화수소수, 알코올 패드\
    #         '근육통': [13],  # 아렉스\
    #         '감기': [1, 3],  # 투즐콜드, 타이레놀\
    #         '가려움 완화': [6],  # 세이프론\
    #         '입안염증 완화': [4],  # 스트렙실\
    #         '상처 치료': [11, 12],  # 밴드, 연고\
    #     }\
    #     ```\
    # 각 번호는 이미지를 바탕으로 한 물품을 가리킵니다."\
    
    
]