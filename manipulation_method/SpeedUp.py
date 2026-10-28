import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
SpeedUp 변조 (랜덤 속도감 효과)

빠르게 달리는 차 안에서 본 풍경처럼, 이미지의 좌우 가장자리로 갈수록
가로 방향 모션 블러를 점점 강하게 넣어 속도감을 만든다.
(자율주행 데이터 증강 라이브러리 Automold의 add_speed와 같은 방식)

1. 가운데의 일정 폭(선명 영역)은 원본 그대로 둔다.
2. 선명 영역 바깥부터 좌우 끝까지 블러 길이를 0에서 최대 길이까지 점점 늘린다.
3. 가로 방향으로만 픽셀을 평균 내는 모션 블러(가로 막대 커널)를 적용한다.
   (블러 길이가 다른 사본 여러 장을 만들고, 세로줄마다 알맞은 사본을 섞는다)

가장자리의 선과 글자는 가로로 길게 끌린 잔상이 되고, 가운데는 선명하게 남는다.

가운데 위치는 이미지마다 랜덤으로 조금씩 옮겨 lv1~lv4에 같이 쓰고,
최대 블러 길이(이미지 가로 길이 대비)와 선명 영역의 폭은 level별 범위 안에서 랜덤으로 정한다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 최대 블러 길이 (가로 길이 대비 / 가로 720px 기준) / 선명 영역 폭 (가로 길이 대비):
level 1: 2~4%   / 약 14~29px  / 50~60%
level 2: 4~7%   / 약 29~50px  / 40~50%
level 3: 7~11%  / 약 50~79px  / 30~40%
level 4: 11~16% / 약 79~115px / 20~30%

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 속도감 설정
# blur:  가장자리에서의 최대 블러 길이 (이미지 가로 길이 대비)
# clear: 가운데 선명 영역의 폭 (이미지 가로 길이 대비)
CONFIGS = {
    1: {"blur": (0.02, 0.04), "clear": (0.50, 0.60)},
    2: {"blur": (0.04, 0.07), "clear": (0.40, 0.50)},
    3: {"blur": (0.07, 0.11), "clear": (0.30, 0.40)},
    4: {"blur": (0.11, 0.16), "clear": (0.20, 0.30)},
}

# 가운데 위치를 옮기는 범위 (이미지 가로 길이 대비 ±)
CENTER_JITTER = 0.08

# 블러 길이를 나누는 단계 수 (클수록 부드럽게 이어지고 느려진다)
BLUR_STEPS = 8

# 가장자리로 갈수록 블러가 세지는 곡선 (1이면 일정하게, 클수록 끝에서 급하게 세짐)
RAMP_POWER = 1.5

# True면 가운데 위치를 이미지마다 한 번만 골라 lv1~lv4에 같이 쓴다.
# False면 level마다 따로 고른다.
SAME_STYLE_ALL_LEVELS = True


def SpeedUp(
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
    if width < 4 or height < 2:
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

    # 가운데 위치는 이미지마다, 블러 길이와 선명 영역 폭은 level마다 결정
    center = width * (0.5 + style_rng.uniform(-CENTER_JITTER, CENTER_JITTER))
    max_length = rng.uniform(*config["blur"]) * width
    clear = rng.uniform(*config["clear"])

    # 세로줄마다 블러 세기(0~1)를 정한다. 선명 영역 안은 0, 좌우 끝은 1
    left_edge = center - clear * width / 2
    right_edge = center + clear * width / 2

    x = np.arange(width, dtype=np.float64) + 0.5
    strength = np.zeros(width, dtype=np.float64)

    left = x < left_edge
    strength[left] = (left_edge - x[left]) / max(left_edge, 1.0)

    right = x > right_edge
    strength[right] = (x[right] - right_edge) / max(width - right_edge, 1.0)

    strength = np.clip(strength, 0.0, 1.0) ** RAMP_POWER

    # 블러 세기를 단계 번호(0~BLUR_STEPS, 소수 포함)로 바꾼다.
    position = strength * BLUR_STEPS

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    result = np.zeros_like(rgb)

    # 단계마다 블러 사본을 만들고, 세로줄마다 가까운 두 단계를 비율대로 섞는다.
    for step in range(BLUR_STEPS + 1):

        weight = np.clip(1.0 - np.abs(position - step), 0.0, 1.0).astype(np.float32)

        if not weight.any():
            continue

        length = int(round(max_length * step / BLUR_STEPS))

        if length <= 1:
            blurred = rgb
        else:
            blurred = cv2.blur(rgb, (length, 1), borderType=cv2.BORDER_REFLECT_101)

        result += blurred * weight[None, :, None]

    SpeedUp = Image.fromarray(
        np.clip(np.rint(result), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        SpeedUp.putalpha(alpha)

    return SpeedUp
