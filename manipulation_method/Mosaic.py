import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
Mosaic 변조 (랜덤 모자이크 타일)

이미지 전체를 크기가 제각각인 직사각형 타일로 나누고,
타일마다 그 안의 평균 색 한 가지로 칠해 모자이크 타일 벽화처럼 만든다.

1. 이미지를 위에서 아래로, 높이가 랜덤한 줄(row)로 나눈다.
2. 각 줄을 왼쪽부터, 너비가 랜덤한 타일로 나눈다.
   (줄마다 타일 경계가 어긋나 벽돌을 쌓은 것처럼 불규칙해진다)
3. 타일마다 원본의 평균 색으로 칠한다.
4. 이미지에 따라 타일 사이에 줄눈(grout)을 넣는다. (검은색 또는 흰색 계열)
   줄눈 두께는 타일 크기에 비례하고, 진하기는 level이 오를수록 진해진다.
   (level 1은 줄눈 없이 타일만 있다)

블록 크기가 일정하고 일부 영역에만 적용되는 Pixelate와 달리,
이미지 전체가 크기가 불규칙한 타일로 바뀐다.

줄눈을 넣을지와 줄눈 색은 이미지마다 랜덤으로 한 번 정해 lv1~lv4에 같이 쓰고,
타일 크기는 level별 범위 안에서 랜덤으로 정한다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 타일 한 변의 길이 범위 (이미지 짧은 변 대비 / 짧은 변 720px 기준) / 줄눈 진하기:
level 1: 0.4~1.2% / 약 3~9px   / 줄눈 없음
level 2: 0.8~2.4% / 약 6~17px  / 35%
level 3: 1.5~4.5% / 약 11~32px / 60%
level 4: 3.0~8.0% / 약 22~58px / 85%

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 모자이크 설정
# tile:  타일 한 변의 길이 범위 (이미지 짧은 변 대비)
# grout: 줄눈 진하기 (0이면 줄눈 없음, 1이면 줄눈 색으로 완전히 덮음)
CONFIGS = {
    1: {"tile": (0.004, 0.012), "grout": 0.00},
    2: {"tile": (0.008, 0.024), "grout": 0.35},
    3: {"tile": (0.015, 0.045), "grout": 0.60},
    4: {"tile": (0.030, 0.080), "grout": 0.85},
}

# 줄눈을 넣을 확률
GROUT_PROB = 0.5

# 줄눈 두께 = 평균 타일 한 변의 길이 x 이 비율 (최소 1px)
GROUT_TILE_RATIO = 0.08

# 줄눈 색 후보
GROUT_COLORS = {
    "DARK": (25, 25, 25),
    "LIGHT": (240, 240, 240),
}

# True면 줄눈 사용 여부와 색을 이미지마다 한 번만 골라 lv1~lv4에 같이 쓴다.
# False면 level마다 따로 고른다.
SAME_STYLE_ALL_LEVELS = True


def Mosaic(
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

    config = CONFIGS[level]
    short_side = min(width, height)

    # 줄눈 사용 여부와 색은 이미지마다 결정
    use_grout = style_rng.random() < GROUT_PROB
    grout_color = np.array(
        GROUT_COLORS[style_rng.choice(list(GROUT_COLORS))], dtype=np.float64
    )

    # 타일 크기 범위 (px)
    tile_min = max(1, int(round(config["tile"][0] * short_side)))
    tile_max = max(tile_min, int(round(config["tile"][1] * short_side)))

    # 줄눈 두께는 타일 크기에 비례, 진하기는 level별로 정한다.
    grout = max(1, int(round((tile_min + tile_max) / 2 * GROUT_TILE_RATIO)))
    grout_opacity = config["grout"] if use_grout else 0.0

    # 타일 크기 난수는 numpy로 만들되, 시드는 rng에서 받아 재현성을 유지한다.
    np_rng = np.random.default_rng(rng.getrandbits(64))

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB"))

    # 직사각형 영역의 합을 빠르게 구하기 위한 누적합 표 (H+1 x W+1 x 3)
    integral = cv2.integral(rgb, sdepth=cv2.CV_64F)

    result = np.empty((height, width, 3), dtype=np.float64)

    top = 0
    while top < height:

        bottom = min(height, top + int(np_rng.integers(tile_min, tile_max + 1)))

        # 줄 안의 타일 경계 (왼쪽 0부터 오른쪽 끝까지)
        widths = np_rng.integers(tile_min, tile_max + 1, size=width // tile_min + 2)
        xs = np.concatenate([[0], np.cumsum(widths)])
        xs = np.append(xs[xs < width], width)

        # 타일마다 평균 색 = 영역의 합 / 넓이
        sums = (
            integral[bottom, xs[1:]] - integral[bottom, xs[:-1]]
            - integral[top, xs[1:]] + integral[top, xs[:-1]]
        )
        tile_widths = np.diff(xs)
        means = sums / ((bottom - top) * tile_widths)[:, None]

        result[top:bottom] = np.repeat(means, tile_widths, axis=0)[None, :, :]

        if grout_opacity > 0:
            # 줄 위쪽 가로 줄눈 (맨 위 줄은 이미지 경계라 생략)
            if top > 0:
                band = result[top:min(bottom, top + grout)]
                band[...] = (1 - grout_opacity) * band + grout_opacity * grout_color

            # 타일 사이 세로 줄눈
            columns = (xs[1:-1, None] + np.arange(grout)[None, :]).ravel()
            columns = columns[columns < width]
            result[top:bottom, columns] = (
                (1 - grout_opacity) * result[top:bottom, columns]
                + grout_opacity * grout_color
            )

        top = bottom

    Mosaic = Image.fromarray(
        np.clip(np.rint(result), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Mosaic.putalpha(alpha)

    return Mosaic
