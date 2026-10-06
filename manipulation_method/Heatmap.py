import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
Heatmap 변조 (랜덤 컬러맵 적용)

이미지를 흑백으로 바꾼 뒤, 밝기 값마다 정해진 색을 입혀
열화상 사진 같은 모습으로 만든다.

1. 이미지를 흑백(밝기 0~255)으로 바꾼다. 원래의 색 정보는 사라진다.
2. 컬러맵(밝기 값 -> 색 대응표)을 하나 랜덤으로 고른다.
3. 각 픽셀의 밝기에 해당하는 색으로 칠한다.

형태와 명암의 순서는 유지되고 색만 완전히 바뀐다.
컬러맵은 level별 후보 중에서 랜덤으로 고른다.
level은 변조의 강도가 아니라 컬러맵의 종류를 나눈 것이다.

출력은 RGB이고(투명도가 있는 이미지는 RGBA), 출력 크기는 원본과 같다.

level별 컬러맵 후보:
level 1: JET, BONE, AUTUMN            (부드러운 계열)
level 2: HOT, PLASMA, VIRIDIS         (대비가 있는 계열)
level 3: RAINBOW, OCEAN, SUMMER       (다채로운 계열)
level 4: HSV, COOL, PINK, MAGMA       (극적인 계열)

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 OpenCV 컬러맵 후보
COLORMAP_POOL = {
    1: [cv2.COLORMAP_JET, cv2.COLORMAP_BONE, cv2.COLORMAP_AUTUMN],
    2: [cv2.COLORMAP_HOT, cv2.COLORMAP_PLASMA, cv2.COLORMAP_VIRIDIS],
    3: [cv2.COLORMAP_RAINBOW, cv2.COLORMAP_OCEAN, cv2.COLORMAP_SUMMER],
    4: [cv2.COLORMAP_HSV, cv2.COLORMAP_COOL, cv2.COLORMAP_PINK, cv2.COLORMAP_MAGMA],
}


def Heatmap(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in COLORMAP_POOL:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # level별 후보 중에서 컬러맵을 랜덤으로 선택
    colormap = rng.choice(COLORMAP_POOL[level])

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    # 흑백(밝기)으로 변환
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 밝기 값마다 컬러맵의 색을 입힌다 (OpenCV의 결과는 BGR 순서).
    colored_bgr = cv2.applyColorMap(gray, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    Heatmap = Image.fromarray(colored_rgb)

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Heatmap.putalpha(alpha)

    return Heatmap