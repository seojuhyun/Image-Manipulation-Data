import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
ErodeDilate 변조 (랜덤 침식 / 팽창)

모폴로지(형태학) 연산으로 이미지의 밝은 영역과 어두운 영역의 경계를 깎거나 불린다.
R, G, B 채널마다, 커널(구조 요소) 모양 안에 들어오는 이웃 픽셀을 보고
- EROSION(침식):  이웃 중 가장 어두운 값으로 바꾼다.
                  어두운 영역이 번져 검은 선과 글자가 굵고 뭉툭해지고,
                  작은 밝은 점(별, 눈 하이라이트)은 사라진다.
- DILATION(팽창): 이웃 중 가장 밝은 값으로 바꾼다.
                  밝은 영역이 번져 검은 선과 글자가 가늘어지다가 끊기고,
                  작은 어두운 점은 사라진다.

커널 모양은 아래 세 가지 중에서 고른다. 모양에 따라 번지는 형태가 달라진다.
- RECT(사각형): 모서리가 네모나게 번진다.
- ELLIPSE(원):  둥글게 번진다.
- CROSS(십자):  가로/세로로만 번져 십자 모양 자국이 남는다.

연산 종류와 커널 모양은 이미지마다 랜덤으로 한 번 정해 lv1~lv4에 같이 쓰고,
커널 크기만 level별 범위 안에서 랜덤으로 정한다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 커널 크기 범위 (이미지 짧은 변 대비 / 짧은 변 720px 기준):
level 1: 0.3~0.45% / 약 2~3px
level 2: 0.55~0.8% / 약 4~6px
level 3: 0.95~1.3% / 약 7~9px
level 4: 1.5~2.2%  / 약 11~16px

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 커널 크기 범위 (이미지 짧은 변 대비)
KERNEL_RANGES = {
    1: (0.003, 0.0045),
    2: (0.0055, 0.008),
    3: (0.0095, 0.013),
    4: (0.015, 0.022),
}

# 연산 종류 후보
OPERATIONS = ["EROSION", "DILATION"]

# 커널 모양 후보
KERNEL_SHAPES = {
    "RECT": cv2.MORPH_RECT,
    "ELLIPSE": cv2.MORPH_ELLIPSE,
    "CROSS": cv2.MORPH_CROSS,
}

# True면 연산 종류와 커널 모양을 이미지마다 한 번만 골라 lv1~lv4에 같이 쓴다. (강도만 level별로 다름)
# False면 level마다 따로 고른다.
SAME_STYLE_ALL_LEVELS = True


def ErodeDilate(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in KERNEL_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # 너무 작은 이미지는 그대로 돌려준다.
    if width < 3 or height < 3:
        return image.copy()

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    # 방식을 고르는 style_rng는 이미지 내용만으로 시드를 만들어
    # lv1~lv4가 같은 방식을 쓰고 강도만 달라지게 한다.
    if rng is None:
        image_seed = zlib.crc32(image.tobytes())
        rng = random.Random(image_seed + level)
        style_rng = random.Random(image_seed) if SAME_STYLE_ALL_LEVELS else rng
    else:
        style_rng = rng

    # 연산 종류와 커널 모양은 이미지마다, 커널 크기는 level마다 랜덤으로 결정
    operation = style_rng.choice(OPERATIONS)
    shape = style_rng.choice(list(KERNEL_SHAPES))

    size = rng.uniform(*KERNEL_RANGES[level]) * min(width, height)
    size = max(2, int(round(size)))

    kernel = cv2.getStructuringElement(KERNEL_SHAPES[shape], (size, size))

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB"))

    if operation == "EROSION":
        morphed = cv2.erode(rgb, kernel)
    else:
        morphed = cv2.dilate(rgb, kernel)

    ErodeDilate = Image.fromarray(morphed)

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        ErodeDilate.putalpha(alpha)

    return ErodeDilate
