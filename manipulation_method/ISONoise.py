import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
ISONoise 변조 (랜덤 카메라 센서 노이즈)

어두운 곳에서 ISO 감도를 높여 찍은 사진처럼 카메라 센서 노이즈를 흉내 낸다.

1. 픽셀값을 빛의 양에 비례하는 값(선형 밝기, 감마 2.2)으로 바꾼다.
2. 두 종류의 노이즈를 더한다.
   - 밝기 노이즈: 빛의 양에 따라 세기가 달라지는 샷 노이즈 + 일정한 읽기 노이즈.
                  세 채널에 같은 값을 더해 밝기만 자글자글하게 흔든다.
   - 색 노이즈:   채널마다 다른 난수를 1~2px 크기로 뭉쳐서 더해,
                  실제 센서처럼 얼룩덜룩한 색 얼룩(chroma noise)을 만든다.
3. 다시 원래 밝기 단위로 되돌린다.

감마 곡선 때문에 노이즈가 어두운 영역과 중간 밝기에서 크게 보이고,
흰 배경처럼 가장 밝은 곳에서는 거의 보이지 않는다.
밝기와 상관없이 고르게 더하는 AddNoise와 다른 점이다.
(albumentations ISONoise의 두 매개변수 intensity, color_shift를 그대로 쓴다)

색 노이즈 비율(color_shift)과 색 얼룩 크기는 이미지마다 랜덤으로 한 번 정해
lv1~lv4에 같이 쓰고, 강도(intensity)만 level별 범위 안에서 랜덤으로 정한다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 intensity 범위:
level 1: 0.5~1.0
level 2: 1.0~1.8
level 3: 1.8~3.0
level 4: 3.0~4.5

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 노이즈 강도(intensity) 범위
INTENSITY_RANGES = {
    1: (0.5, 1.0),
    2: (1.0, 1.8),
    3: (1.8, 3.0),
    4: (3.0, 4.5),
}

# 밝기 노이즈 대비 색 노이즈의 비율 범위
COLOR_SHIFT_RANGE = (0.4, 1.0)

# intensity 1일 때의 노이즈 세기 (선형 밝기 0~1 기준)
# 샷 노이즈는 sqrt(밝기)에 비례하고, 읽기 노이즈는 밝기와 상관없이 일정하다.
LUMA_SHOT = 0.035
LUMA_READ = 0.003
CHROMA_SHOT = 0.05
CHROMA_READ = 0.004

# 색 얼룩 크기 (가우시안 sigma, px)
CHROMA_GRAIN_RANGE = (0.6, 1.4)

GAMMA = 2.2

# True면 색 노이즈 비율과 얼룩 크기를 이미지마다 한 번만 골라 lv1~lv4에 같이 쓴다.
# False면 level마다 따로 고른다.
SAME_STYLE_ALL_LEVELS = True


def ISONoise(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in INTENSITY_RANGES:
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

    # 색 노이즈 비율과 얼룩 크기는 이미지마다, 강도는 level마다 랜덤으로 결정
    color_shift = style_rng.uniform(*COLOR_SHIFT_RANGE)
    grain = style_rng.uniform(*CHROMA_GRAIN_RANGE)
    intensity = rng.uniform(*INTENSITY_RANGES[level])

    # 픽셀 단위 난수는 numpy로 만들되, 시드는 rng에서 받아 재현성을 유지한다.
    np_rng = np.random.default_rng(rng.getrandbits(64))

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    height, width = rgb.shape[:2]

    # 1. 선형 밝기로 바꾼다.
    linear = rgb ** GAMMA
    luminance = linear.mean(axis=2, keepdims=True)
    shot = np.sqrt(luminance)

    # 2-1. 밝기 노이즈 (세 채널 공통)
    luma_std = intensity * (LUMA_SHOT * shot + LUMA_READ)
    luma_noise = np_rng.standard_normal((height, width, 1), dtype=np.float32)

    # 2-2. 색 노이즈 (채널마다 다른 난수를 grain 크기로 뭉친다)
    chroma_noise = np_rng.standard_normal((height, width, 3), dtype=np.float32)
    chroma_noise = cv2.GaussianBlur(chroma_noise, (0, 0), sigmaX=grain)
    chroma_noise /= max(float(chroma_noise.std()), 1e-6)

    chroma_std = intensity * color_shift * (CHROMA_SHOT * shot + CHROMA_READ)

    noisy = linear + luma_noise * luma_std + chroma_noise * chroma_std

    # 3. 원래 밝기 단위로 되돌린다.
    restored = np.clip(noisy, 0.0, 1.0) ** (1.0 / GAMMA)

    ISONoise = Image.fromarray(
        np.clip(np.rint(restored * 255.0), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        ISONoise.putalpha(alpha)

    return ISONoise
