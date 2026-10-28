import random
import zlib

import cv2
import numpy as np
from PIL import Image, ImageFilter

"""
EdgeDetect 변조 (랜덤 경계 검출)

서로 다른 필터 중 하나를 랜덤으로 골라 이미지의 경계(edge)를 검출하고,
검출 결과(edge map)를 level별 비율로 원본 위에 덮는다.

경계 검출 필터는 아래 다섯 가지 중에서 랜덤으로 고른다.
- FIND_EDGES: 흑백으로 바꾼 뒤 3x3 라플라시안 계열 필터를 적용한다.
              검은 바탕에 흰 경계선이 남는다.
- CANNY:      Canny 검출기로 얇고 끊김 없는 흰 경계선만 남긴다.
              (두 임계값은 랜덤, 결과는 흑백 이진 이미지)
- SOBEL:      가로/세로 밝기 변화량(그래디언트)의 크기를 밝기로 나타낸다.
              경계가 강할수록 밝고, 두꺼운 선은 이중선으로 보인다.
- LAPLACIAN:  R, G, B 채널마다 3x3 라플라시안을 적용한다.
              검은 바탕에 원래 색이 묻은 경계선이 남는다.
- CONTOUR:    PIL의 윤곽선 필터로, 흰 바탕에 어두운 윤곽선만 남긴다.
              (연필 스케치 같은 느낌)

덮는 비율이 100%면 경계 검출 결과만 남고, 낮을수록 원본이 비쳐 보인다.
필터 종류는 이미지마다 랜덤으로 한 번 정해 lv1~lv4에 같이 쓰고,
덮는 비율만 level별 범위 안에서 랜덤으로 정한다.
출력은 RGB이고(투명도가 있는 이미지는 RGBA), 출력 크기는 원본과 같다.

level별 경계 검출 결과를 덮는 비율:
level 1: 20~35%
level 2: 40~55%
level 3: 60~75%
level 4: 85~100% (거의 경계 검출 결과만 남음)

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 경계 검출 결과를 덮는 비율 범위
BLEND_RANGES = {
    1: (0.20, 0.35),
    2: (0.40, 0.55),
    3: (0.60, 0.75),
    4: (0.85, 1.00),
}

# 경계 검출 필터 후보
DETECTORS = ["FIND_EDGES", "CANNY", "SOBEL", "LAPLACIAN", "CONTOUR"]

# Canny의 낮은 임계값 범위 (높은 임계값은 낮은 값의 2~3배)
CANNY_LOW_RANGE = (40, 100)

# True면 경계 검출 필터와 Canny 임계값을(를) 이미지마다 한 번만 골라 lv1~lv4에 같이 쓴다. (강도만 level별로 다름)
# False면 level마다 따로 고른다.
SAME_STYLE_ALL_LEVELS = True


def edge_map(rgb, detector, rng):
    # 선택한 필터로 경계 검출 결과를 만든다. (H x W x 3, uint8)

    if detector == "FIND_EDGES":
        gray = Image.fromarray(rgb).convert("L")
        edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES))

    elif detector == "CANNY":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        low = rng.uniform(*CANNY_LOW_RANGE)
        high = low * rng.uniform(2.0, 3.0)
        edges = cv2.Canny(gray, low, high)

    elif detector == "SOBEL":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        # 3x3 Sobel의 최대 응답(약 4 x 255)을 0~255 범위로 맞춘다.
        magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y) / 4.0
        edges = np.clip(magnitude, 0, 255).astype(np.uint8)

    elif detector == "LAPLACIAN":
        kernel = np.array(
            [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]],
            dtype=np.float32
        )
        filtered = cv2.filter2D(rgb.astype(np.float32), -1, kernel)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    else:  # CONTOUR
        contour = Image.fromarray(rgb).filter(ImageFilter.CONTOUR)
        return np.asarray(contour)

    return np.repeat(edges[..., None], 3, axis=2)


def EdgeDetect(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in BLEND_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # 너무 작은 이미지는 경계를 검출할 수 없으므로 그대로 돌려준다.
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

    # 경계 검출 필터는 이미지마다, 덮는 비율은 level마다 랜덤으로 결정
    detector = style_rng.choice(DETECTORS)
    blend = rng.uniform(*BLEND_RANGES[level])

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB"))
    edges = edge_map(rgb, detector, style_rng)

    # level별 비율로 경계 검출 결과를 원본 위에 덮는다.
    mixed = (1.0 - blend) * rgb.astype(np.float32) + blend * edges.astype(np.float32)

    EdgeDetect = Image.fromarray(
        np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        EdgeDetect.putalpha(alpha)

    return EdgeDetect
