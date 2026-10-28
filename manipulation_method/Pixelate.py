import math
import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
Pixelate 변조 (랜덤 일부 영역 픽셀화)

이미지의 일부 영역을 랜덤으로 골라 큰 네모 픽셀(블록)로 바꾼다.
얼굴이나 글자를 가리는 모자이크 처리와 같은 모습이다.

1. 사각형 또는 타원 모양의 영역 1~3개를 랜덤한 위치와 가로세로 비율로 잡는다.
   (영역 넓이의 합 = 이미지 넓이 x level별 비율)
2. 영역 안을 블록 크기만큼씩 나눠, 블록마다 그 안의 평균 색으로 칠한다.

영역 밖은 원본 그대로이고, 영역의 경계는 선명하게 끊긴다.
이미지 전체를 크기가 불규칙한 타일로 바꾸는 Mosaic과 달리 "일부 영역"만 바뀐다.

영역의 개수, 중심 위치, 모양, 가로세로 비율은 이미지마다 랜덤으로 한 번 정해
lv1~lv4에 같이 쓴다. level이 오를수록 같은 자리의 영역이 넓어지고 블록도 커진다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 영역 넓이 비율 (이미지 넓이 대비) / 블록 크기 (짧은 변 대비 / 720px 기준):
level 1: 5~10%  / 1.0~1.5% (약 7~11px)
level 2: 10~20% / 1.5~2.5% (약 11~18px)
level 3: 20~35% / 2.5~4.0% (약 18~29px)
level 4: 35~55% / 4.0~6.0% (약 29~43px)

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 픽셀화 설정
# area:  픽셀화할 영역 넓이의 합 (이미지 넓이 대비)
# block: 블록 한 변의 길이 (이미지 짧은 변 대비)
CONFIGS = {
    1: {"area": (0.05, 0.10), "block": (0.010, 0.015)},
    2: {"area": (0.10, 0.20), "block": (0.015, 0.025)},
    3: {"area": (0.20, 0.35), "block": (0.025, 0.040)},
    4: {"area": (0.35, 0.55), "block": (0.040, 0.060)},
}

# 픽셀화할 영역의 개수 범위
REGION_COUNT_RANGE = (1, 3)

# 영역의 가로/세로 비율 범위
ASPECT_RANGE = (1 / 2, 2)

# 영역 모양 후보
SHAPES = ["RECT", "ELLIPSE"]

# True면 영역의 개수, 위치, 모양을 이미지마다 한 번만 골라 lv1~lv4에 같이 쓴다.
# False면 level마다 따로 고른다.
SAME_STYLE_ALL_LEVELS = True


def pixelate_patch(patch, block):
    # patch를 block x block 단위로 나눠 블록마다 평균 색으로 칠한다.
    height, width = patch.shape[:2]
    rows = math.ceil(height / block)
    cols = math.ceil(width / block)

    # 블록 크기로 나누어떨어지도록 가장자리를 늘린다.
    padded = cv2.copyMakeBorder(
        patch, 0, rows * block - height, 0, cols * block - width, cv2.BORDER_REPLICATE
    )

    means = padded.reshape(rows, block, cols, block, 3).mean(axis=(1, 3))
    blocks = np.repeat(np.repeat(means, block, axis=0), block, axis=1)

    return blocks[:height, :width]


def Pixelate(
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

    # 영역의 개수, 중심 위치, 가로세로 비율, 넓이 배분, 모양은 이미지마다 결정
    regions = []
    for _ in range(style_rng.randint(*REGION_COUNT_RANGE)):
        regions.append({
            "cx": style_rng.random(),
            "cy": style_rng.random(),
            "aspect": math.exp(
                style_rng.uniform(math.log(ASPECT_RANGE[0]), math.log(ASPECT_RANGE[1]))
            ),
            "weight": style_rng.uniform(0.5, 1.0),
            "shape": style_rng.choice(SHAPES),
        })

    # 영역 넓이와 블록 크기는 level마다 결정
    area_ratio = rng.uniform(*config["area"])
    block = max(2, int(round(rng.uniform(*config["block"]) * short_side)))

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB"))
    result = rgb.astype(np.float32)

    total_weight = sum(region["weight"] for region in regions)

    for region in regions:

        area = width * height * area_ratio * region["weight"] / total_weight
        region_w = int(round(math.sqrt(area * region["aspect"])))
        region_h = int(round(math.sqrt(area / region["aspect"])))
        region_w = max(1, min(region_w, width))
        region_h = max(1, min(region_h, height))

        # 중심 위치에 맞추되 이미지 밖으로 나가지 않게 한다.
        left = int(round(region["cx"] * width - region_w / 2))
        top = int(round(region["cy"] * height - region_h / 2))
        left = min(max(0, left), width - region_w)
        top = min(max(0, top), height - region_h)

        pixelated = pixelate_patch(
            rgb[top:top + region_h, left:left + region_w], block
        )

        target = result[top:top + region_h, left:left + region_w]

        if region["shape"] == "RECT":
            target[...] = pixelated
        else:
            mask = np.zeros((region_h, region_w), dtype=np.uint8)
            cv2.ellipse(
                mask,
                (region_w // 2, region_h // 2),
                (max(1, region_w // 2), max(1, region_h // 2)),
                0, 0, 360, 1, -1
            )
            inside = mask.astype(bool)
            target[inside] = pixelated[inside]

    Pixelate = Image.fromarray(
        np.clip(np.rint(result), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Pixelate.putalpha(alpha)

    return Pixelate
