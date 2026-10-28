import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
Filtering 변조 (랜덤 필터링)

가우시안 블러(Blur) 이외의 고전적인 공간 필터 중 하나를 랜덤으로 골라 적용한다.
조작(forgery) 흔적을 감추기 위해 추가로 적용하는 후처리(anti-forensics)에서
자주 쓰이는 필터들이다.

- MEDIAN:    커널 안 픽셀값의 중앙값으로 바꾼다.
             잡음과 가는 선, 작은 점이 사라지고 모서리가 둥글게 뭉개진다.
- BILATERAL: 가까우면서 색도 비슷한 픽셀끼리만 평균 낸다.
             경계는 살리고 면은 매끈하게 만들어, 반복할수록
             셀 애니메이션처럼 평평하고 질감 없는 그림이 된다.
- BOX:       커널 안 픽셀값의 단순 평균으로 바꾼다.
             가우시안 블러와 달리 경계가 네모난 모양으로 번진다.

필터 종류는 이미지마다 랜덤으로 한 번 정해 lv1~lv4에 같이 쓰고,
커널 크기(BILATERAL은 색 범위와 반복 횟수도)만 level별로 달라진다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 커널 크기 범위 (이미지 짧은 변 대비 / 짧은 변 720px 기준):
level 1: 0.4~0.6% / 약 3~4px
level 2: 0.7~1.0% / 약 5~7px
level 3: 1.2~1.7% / 약 9~12px
level 4: 2.0~2.8% / 약 14~20px
(BILATERAL은 커널 최대 15px, 색 범위 30/50/75/100, 반복 1/2/3/4회)

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 필터 설정
# kernel:      커널 크기 범위 (이미지 짧은 변 대비)
# sigma_color: BILATERAL에서 "비슷한 색"으로 볼 색 차이 범위 (클수록 더 많이 섞임)
# iterations:  BILATERAL 반복 횟수
CONFIGS = {
    1: {"kernel": (0.004, 0.006), "sigma_color": 30, "iterations": 1},
    2: {"kernel": (0.007, 0.010), "sigma_color": 50, "iterations": 2},
    3: {"kernel": (0.012, 0.017), "sigma_color": 75, "iterations": 3},
    4: {"kernel": (0.020, 0.028), "sigma_color": 100, "iterations": 4},
}

# 필터 종류 후보
FILTERS = ["MEDIAN", "BILATERAL", "BOX"]

# BILATERAL은 커널이 크면 매우 느려지므로 크기를 제한하고 반복 횟수로 세기를 낸다.
BILATERAL_MAX_KERNEL = 15

# True면 필터 종류를 이미지마다 한 번만 골라 lv1~lv4에 같이 쓴다. (강도만 level별로 다름)
# False면 level마다 따로 고른다.
SAME_STYLE_ALL_LEVELS = True


def Filtering(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in CONFIGS:
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

    config = CONFIGS[level]

    # 필터 종류는 이미지마다, 커널 크기는 level마다 랜덤으로 결정
    filter_name = style_rng.choice(FILTERS)

    size = rng.uniform(*config["kernel"]) * min(width, height)
    size = max(3, int(round(size)))

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB"))

    if filter_name == "MEDIAN":
        # 중앙값 필터는 커널 크기가 홀수여야 한다.
        if size % 2 == 0:
            size += 1
        filtered = cv2.medianBlur(rgb, size)

    elif filter_name == "BILATERAL":
        diameter = min(size, BILATERAL_MAX_KERNEL)
        filtered = rgb
        for _ in range(config["iterations"]):
            filtered = cv2.bilateralFilter(
                filtered,
                diameter,
                config["sigma_color"],
                diameter
            )

    else:  # BOX
        filtered = cv2.blur(
            rgb,
            (size, size),
            borderType=cv2.BORDER_REFLECT_101
        )

    Filtering = Image.fromarray(filtered)

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Filtering.putalpha(alpha)

    return Filtering
