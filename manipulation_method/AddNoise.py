import random
import zlib

import numpy as np
from PIL import Image

"""
AddNoise 변조 (랜덤 가우시안 노이즈 추가)

모든 픽셀에 평균 0, 표준편차 sigma인 정규분포(가우시안) 난수를 더한다.
이미지 전체에 모래알 같은 자글자글한 잡음이 고르게 생기며,
흰 배경과 검은 선에도 같은 세기로 생긴다.

노이즈 방식은 아래 두 가지 중에서 랜덤으로 고른다.
- COLOR: R, G, B 채널마다 서로 다른 난수를 더한다. 알록달록한 색 잡음이 생긴다.
- MONO:  세 채널에 같은 난수를 더한다. 색은 그대로이고 밝기만 흔들리는 흑백 잡음이 생긴다.

밝기와 상관없이 같은 세기로 더해진다는 점이, 어두운 곳에서 강하고
색 얼룩이 뭉쳐 보이는 카메라 센서 노이즈(ISONoise)와 다르다.

노이즈 방식은 이미지마다 랜덤으로 한 번 정해 lv1~lv4에 같이 쓰고,
sigma만 level별 범위 안에서 랜덤으로 정한다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 sigma 범위 (픽셀값 0~255 기준):
level 1: 4~8
level 2: 8~15
level 3: 15~28
level 4: 28~45

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 노이즈 표준편차(sigma) 범위
SIGMA_RANGES = {
    1: (4, 8),
    2: (8, 15),
    3: (15, 28),
    4: (28, 45),
}

# 노이즈 방식 후보
NOISE_TYPES = ["COLOR", "MONO"]

# True면 노이즈 방식을 이미지마다 한 번만 골라 lv1~lv4에 같이 쓴다. (강도만 level별로 다름)
# False면 level마다 따로 고른다.
SAME_STYLE_ALL_LEVELS = True


def AddNoise(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in SIGMA_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    # 방식을 고르는 style_rng는 이미지 내용만으로 시드를 만들어
    # lv1~lv4가 같은 방식을 쓰고 강도만 달라지게 한다.
    if rng is None:
        image_seed = zlib.crc32(image.tobytes())
        rng = random.Random(image_seed + level)
        style_rng = random.Random(image_seed) if SAME_STYLE_ALL_LEVELS else rng
    else:
        style_rng = rng

    # 노이즈 방식은 이미지마다, sigma는 level마다 랜덤으로 결정
    noise_type = style_rng.choice(NOISE_TYPES)
    sigma = rng.uniform(*SIGMA_RANGES[level])

    # 픽셀 단위 난수는 numpy로 만들되, 시드는 rng에서 받아 재현성을 유지한다.
    np_rng = np.random.default_rng(rng.getrandbits(64))

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    height, width = rgb.shape[:2]

    if noise_type == "COLOR":
        noise = np_rng.standard_normal((height, width, 3), dtype=np.float32)
    else:
        noise = np_rng.standard_normal((height, width, 1), dtype=np.float32)

    noisy = rgb + noise * sigma

    AddNoise = Image.fromarray(
        np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        AddNoise.putalpha(alpha)

    return AddNoise
