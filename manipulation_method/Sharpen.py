import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
Sharpen 변조 (랜덤 언샤프 마스크)

이미지의 경계(edge) 대비를 강하게 만들어 날카로워 보이게 한다.
포토샵의 언샤프 마스크(Unsharp Mask)와 같은 방식이다.

1. 이미지를 가우시안으로 살짝 흐린 사본을 만든다.
2. 원본 - 흐린 사본 = 경계 부분에만 값이 남는 "디테일" 성분이다.
3. 원본에 디테일 성분을 amount배만큼 더한다.
       출력 = 원본 + amount x (원본 - 흐린 사본)

경계의 밝은 쪽은 더 밝게, 어두운 쪽은 더 어둡게 밀리므로
선과 글자의 테두리가 또렷해진다.
amount가 커질수록 검은 선 바깥에 흰 테두리(헤일로)가 생기고,
그러데이션과 스크린톤의 질감이 거칠게 도드라진다.

흐림 반경은 이미지마다 랜덤으로 한 번 정해 lv1~lv4에 같이 쓰고,
amount(강도)만 level별 범위 안에서 랜덤으로 정한다.
흐림 반경이 작으면 픽셀 단위의 잔 경계가, 크면 굵은 윤곽이 강조된다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 amount 범위:
level 1: 0.6~1.0배
level 2: 1.0~1.8배
level 3: 1.8~3.0배
level 4: 3.0~5.0배

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 디테일 성분을 더하는 배율(amount) 범위
AMOUNT_RANGES = {
    1: (0.6, 1.0),
    2: (1.0, 1.8),
    3: (1.8, 3.0),
    4: (3.0, 5.0),
}

# 흐린 사본을 만들 때의 가우시안 sigma 범위
# (이미지 짧은 변 대비 / 짧은 변 720px 기준 약 0.9~2.2px)
RADIUS_RANGE = (0.0012, 0.003)

# True면 흐림 반경을(를) 이미지마다 한 번만 골라 lv1~lv4에 같이 쓴다. (강도만 level별로 다름)
# False면 level마다 따로 고른다.
SAME_STYLE_ALL_LEVELS = True


def Sharpen(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in AMOUNT_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # 너무 작은 이미지는 그대로 돌려준다.
    if width < 2 or height < 2:
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

    # 흐림 반경은 이미지마다, 강도는 level마다 랜덤으로 결정
    sigma = max(0.5, style_rng.uniform(*RADIUS_RANGE) * min(width, height))
    amount = rng.uniform(*AMOUNT_RANGES[level])

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB")).astype(np.float32)

    blurred = cv2.GaussianBlur(
        rgb,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101
    )

    # 원본에 디테일 성분(원본 - 흐린 사본)을 amount배 더한다.
    sharpened = rgb + amount * (rgb - blurred)

    Sharpen = Image.fromarray(
        np.clip(np.rint(sharpened), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Sharpen.putalpha(alpha)

    return Sharpen
