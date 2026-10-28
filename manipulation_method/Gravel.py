import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
Gravel 변조 (랜덤 자갈 효과)

이미지 곳곳에 가로로 긴 직사각형 "자갈 밭" 패치를 만들고,
패치 안을 밝기가 제각각인 작은 네모 알갱이로 촘촘히 덮는다.
(도로 위 자갈을 흉내 내는 Automold의 add_gravel과 같은 방식)

1. 가로로 긴 직사각형 패치를 랜덤한 위치와 크기로 여러 개 놓는다.
2. 각 패치 넓이의 1/10 개수만큼 1~4px 크기의 알갱이를 랜덤한 위치에 찍는다.
3. 알갱이 자리의 밝기(HLS 색 공간의 L)만 0~255 중 랜덤한 값으로 바꾼다.
   색상(H)과 채도(S)는 원래 값이므로, 흰 배경 위에서는 흑백 알갱이가,
   색이 있는 곳에서는 같은 색의 밝고 어두운 알갱이가 생긴다.

원본은 도로가 있는 이미지 아래쪽 1/4에만 넣지만,
웹툰 컷에는 도로가 없으므로 이미지 전체에 고르게 넣는다.

패치가 덮는 넓이는 level별 범위 안에서 랜덤으로 정한다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 자갈 패치가 덮는 넓이 비율 (이미지 넓이 대비):
level 1: 1.0~2.5%
level 2: 2.5~5%
level 3: 5~10%
level 4: 10~18%

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 자갈 패치가 덮는 넓이 비율 범위
COVERAGE_RANGES = {
    1: (0.010, 0.025),
    2: (0.025, 0.050),
    3: (0.050, 0.100),
    4: (0.100, 0.180),
}

# 패치의 가로/세로 길이 범위 (이미지 짧은 변 대비 / 720px 기준 약 58~200px, 11~32px)
PATCH_WIDTH_RANGE = (0.08, 0.28)
PATCH_HEIGHT_RANGE = (0.015, 0.045)

# 알갱이 한 변의 길이 범위 (이미지 짧은 변 대비 / 720px 기준 약 1~4px)
GRAIN_SIZE_RANGE = (0.0014, 0.0056)

# 패치 넓이 1px당 알갱이 수 (원본 Automold: 넓이 // 10)
GRAIN_DENSITY = 0.1

# 패치 개수 상한 (아주 큰 이미지에서 시간이 너무 걸리지 않도록)
MAX_PATCHES = 3000


def Gravel(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in COVERAGE_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # 너무 작은 이미지는 그대로 돌려준다.
    if width < 4 or height < 4:
        return image.copy()

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    short_side = min(width, height)

    # level별 범위 안에서 덮을 넓이를 랜덤으로 결정
    coverage = rng.uniform(*COVERAGE_RANGES[level])
    target_area = width * height * coverage

    # 알갱이 난수는 numpy로 만들되, 시드는 rng에서 받아 재현성을 유지한다.
    np_rng = np.random.default_rng(rng.getrandbits(64))

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB"))

    # ------------------------------------------------------------
    # 1~2. 패치를 놓고 패치마다 알갱이 위치, 크기, 밝기를 정한다.
    # ------------------------------------------------------------
    grain_x, grain_y, grain_size, grain_value = [], [], [], []
    covered = 0
    patches = 0

    while covered < target_area and patches < MAX_PATCHES:

        patch_w = int(round(rng.uniform(*PATCH_WIDTH_RANGE) * short_side))
        patch_h = int(round(rng.uniform(*PATCH_HEIGHT_RANGE) * short_side))
        patch_w = max(1, min(patch_w, width))
        patch_h = max(1, min(patch_h, height))

        left = rng.randint(0, width - patch_w)
        top = rng.randint(0, height - patch_h)

        count = max(1, int(patch_w * patch_h * GRAIN_DENSITY))

        grain_x.append(np_rng.integers(left, left + patch_w, size=count))
        grain_y.append(np_rng.integers(top, top + patch_h, size=count))
        grain_size.append(
            np.maximum(
                1,
                np.rint(np_rng.uniform(*GRAIN_SIZE_RANGE, size=count) * short_side)
            ).astype(np.int64)
        )
        grain_value.append(np_rng.integers(0, 256, size=count))

        covered += patch_w * patch_h
        patches += 1

    grain_x = np.concatenate(grain_x)
    grain_y = np.concatenate(grain_y)
    grain_size = np.concatenate(grain_size)
    grain_value = np.concatenate(grain_value).astype(np.uint8)

    # ------------------------------------------------------------
    # 3. 알갱이 자리의 밝기(L)만 바꾼다.
    # ------------------------------------------------------------
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    lightness = hls[..., 1].copy()
    touched = np.zeros((height, width), dtype=bool)

    # 같은 크기의 알갱이끼리 묶어서 한 번에 칠한다.
    for size in np.unique(grain_size):
        selected = grain_size == size
        xs = grain_x[selected]
        ys = grain_y[selected]
        values = grain_value[selected]

        for dy in range(int(size)):
            for dx in range(int(size)):
                yy = np.minimum(ys + dy, height - 1)
                xx = np.minimum(xs + dx, width - 1)
                lightness[yy, xx] = values
                touched[yy, xx] = True

    hls[..., 1] = lightness
    gravel = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    # 알갱이가 없는 곳은 색 공간 변환 오차 없이 원본 그대로 둔다.
    result = rgb.copy()
    result[touched] = gravel[touched]

    Gravel = Image.fromarray(result)

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Gravel.putalpha(alpha)

    return Gravel
