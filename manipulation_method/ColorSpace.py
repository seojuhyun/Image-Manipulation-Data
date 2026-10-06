import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
ColorSpace 변조 (랜덤 색 공간 변환)

이미지를 RGB가 아닌 다른 색 공간(HSV, HLS, YCrCb, LAB, LUV, XYZ)으로
변환한 뒤, 변환된 세 채널의 값을 그대로 R, G, B로 간주해 저장한다.

예를 들어 HSV로 변환하면 세 채널이 (색상, 채도, 밝기)가 되는데,
이것을 (R, G, B)로 읽으므로 색상 값이 빨강의 세기로,
채도 값이 초록의 세기로 나타난다.
그래서 형태와 윤곽은 유지되지만 색이 전혀 다른 이미지가 된다.

색 공간은 level별 후보 중에서 랜덤으로 고른다.
level은 변조의 강도가 아니라 색 공간의 종류를 나눈 것이다.

출력은 RGB이고(투명도가 있는 이미지는 RGBA), 출력 크기는 원본과 같다.

level별 색 공간 후보:
level 1: HSV, HLS
level 2: YCrCb, LAB
level 3: LUV, XYZ
level 4: 위 여섯 가지 전체

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 색 공간 변환 후보
COLORSPACE_POOL = {
    1: [cv2.COLOR_RGB2HSV, cv2.COLOR_RGB2HLS],
    2: [cv2.COLOR_RGB2YCrCb, cv2.COLOR_RGB2LAB],
    3: [cv2.COLOR_RGB2LUV, cv2.COLOR_RGB2XYZ],
    4: [
        cv2.COLOR_RGB2HSV,
        cv2.COLOR_RGB2HLS,
        cv2.COLOR_RGB2YCrCb,
        cv2.COLOR_RGB2LAB,
        cv2.COLOR_RGB2LUV,
        cv2.COLOR_RGB2XYZ,
    ],
}


def ColorSpace(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in COLORSPACE_POOL:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # level별 후보 중에서 색 공간을 랜덤으로 선택
    conversion = rng.choice(COLORSPACE_POOL[level])

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB"))

    # 선택된 색 공간으로 변환한다.
    converted = cv2.cvtColor(rgb, conversion)

    # 변환된 세 채널을 그대로 R, G, B로 간주해 이미지로 만든다.
    ColorSpace = Image.fromarray(converted)

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        ColorSpace.putalpha(alpha)

    return ColorSpace