import math
import random
import zlib

import numpy as np
from PIL import Image

"""
Moire 변조 (랜덤 모아레 무늬)

모니터 화면을 카메라로 찍었을 때처럼, 촘촘한 줄무늬 두 개가 겹치며 생기는
물결 모양 간섭무늬(모아레)를 이미지 위에 입힌다.

1. 아주 촘촘한 줄무늬 두 개를 만든다. 두 줄무늬는 간격이나 각도, 중심이 살짝 다르다.
2. 두 줄무늬를 곱하면, 촘촘한 무늬 위에 굵고 느린 띠(맥놀이)가 나타난다. -> 모아레
3. 모양은 아래 세 가지 중에서 랜덤으로 고른다.
   - LINEAR:   곧은 줄무늬 두 개           -> 곧은 띠 모양 모아레
   - CIRCULAR: 중심이 다른 동심원 무늬 두 개 -> 휘어진 띠 모양 모아레
   - WAVY:     좌표를 사인파로 휘게 한 줄무늬 -> 구불구불한 띠 모양 모아레
4. 이미지마다 랜덤으로 무지개색 모아레(R, G, B 띠의 위치를 조금씩 어긋나게 함)
   또는 흑백 모아레를 고른다.
5. 무늬를 strength만큼 이미지 밝기에 더하고 뺀다. (평균 밝기는 그대로)

모양, 색, 줄무늬 간격과 각도는 이미지마다 랜덤으로 한 번 정해 lv1~lv4에 같이 쓰고,
무늬의 진하기(strength)만 level별 범위 안에서 랜덤으로 정한다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 무늬 진하기 (픽셀값 0~255 중 더하고 빼는 최대 비율):
level 1: 6~10%
level 2: 10~16%
level 3: 16~24%
level 4: 24~34%

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 무늬 진하기 범위
STRENGTH_RANGES = {
    1: (0.06, 0.10),
    2: (0.10, 0.16),
    3: (0.16, 0.24),
    4: (0.24, 0.34),
}

# 모양 후보
PATTERNS = ["LINEAR", "CIRCULAR", "WAVY"]

# 촘촘한 줄무늬의 촘촘함 (짧은 변 하나에 들어가는 줄 수 / 720px 기준 줄 간격 약 2.4~4.8px)
GRATING_RANGE = (150, 300)

# 굵은 띠(맥놀이)의 수 (짧은 변 하나에 보이는 띠 수)
BEAT_RANGE = (4, 15)

# 무지개색 모아레를 쓸 확률
COLOR_PROB = 0.5

# 무늬 중 촘촘한 줄무늬의 비중 (나머지는 굵은 띠)
FINE_WEIGHT = 0.3

# True면 모양, 색, 줄무늬 간격과 각도를 이미지마다 한 번만 골라 lv1~lv4에 같이 쓴다.
# False면 level마다 따로 고른다.
SAME_STYLE_ALL_LEVELS = True


def Moire(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in STRENGTH_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # 너무 작은 이미지는 그대로 돌려준다.
    if width < 4 or height < 4:
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

    short_side = min(width, height)

    # 모양, 색, 줄무늬 간격, 띠 수, 방향은 이미지마다 결정
    pattern = style_rng.choice(PATTERNS)
    colored = style_rng.random() < COLOR_PROB
    grating = style_rng.uniform(*GRATING_RANGE)
    beat = style_rng.uniform(*BEAT_RANGE)
    angle = style_rng.uniform(0, math.pi)

    # 진하기는 level마다 결정
    strength = rng.uniform(*STRENGTH_RANGES[level])

    # 짧은 변을 1로 둔 좌표
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32) / short_side

    cos_a, sin_a = math.cos(angle), math.sin(angle)
    u = xx * cos_a + yy * sin_a      # 줄무늬에 수직인 방향
    v = -xx * sin_a + yy * cos_a     # 줄무늬를 따라가는 방향

    if pattern == "LINEAR":
        # 간격이 조금 다르고 아주 살짝 기운 곧은 줄무늬 두 개
        tilt = style_rng.uniform(-0.02, 0.02)
        u2 = xx * math.cos(angle + tilt) + yy * math.sin(angle + tilt)
        phase1 = grating * u
        phase2 = (grating + beat) * u2

    elif pattern == "CIRCULAR":
        # 중심이 조금 다른 동심원 두 개 (중심은 이미지 밖일 수도 있다)
        cx = style_rng.uniform(-0.5, 1.5) * width / short_side
        cy = style_rng.uniform(-0.5, 1.5) * height / short_side
        shift = beat / grating
        direction = style_rng.uniform(0, 2 * math.pi)
        r1 = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r2 = np.sqrt(
            (xx - cx - shift * math.cos(direction)) ** 2
            + (yy - cy - shift * math.sin(direction)) ** 2
        )
        phase1 = grating * r1
        phase2 = grating * r2

    else:  # WAVY
        # 줄무늬 하나를 사인파로 휘게 만든다.
        amplitude = style_rng.uniform(0.01, 0.03)
        waves = style_rng.uniform(1.0, 3.0)
        offset = style_rng.uniform(0, 2 * math.pi)
        warp = amplitude * np.sin(2 * math.pi * waves * v + offset)
        phase1 = grating * (u + warp)
        phase2 = (grating + beat) * u

    # 두 줄무늬의 곱 = 굵은 띠(차이 주파수) + 촘촘한 무늬(합 주파수)
    difference = 2 * math.pi * (phase1 - phase2)
    total = 2 * math.pi * (phase1 + phase2)

    if colored:
        channel_offsets = [0.0, 2 * math.pi / 3, 4 * math.pi / 3]
    else:
        channel_offsets = [0.0, 0.0, 0.0]

    fine = FINE_WEIGHT * np.cos(total)
    moire = np.stack(
        [(1 - FINE_WEIGHT) * np.cos(difference + off) + fine for off in channel_offsets],
        axis=-1
    )

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB")).astype(np.float32)

    # 무늬를 strength만큼 밝기에 더하고 뺀다.
    result = rgb + moire * (strength * 255.0)

    Moire = Image.fromarray(
        np.clip(np.rint(result), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Moire.putalpha(alpha)

    return Moire
