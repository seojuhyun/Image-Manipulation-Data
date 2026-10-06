import math
import random
import zlib

from PIL import Image

"""
ResizeCrop 변조 (랜덤 크롭 + 리사이즈)

이미지 내 랜덤한 위치에서 level별 비율만큼의 영역을 잘라낸 뒤,
지정한 크기(size)로 resize한다.
size를 주지 않으면 원래 이미지 크기로 되돌린다.

크롭 면적은 level로 고정이고, 위치와 종횡비는 랜덤이다.
종횡비가 원본과 달라지는 만큼 resize 과정에서
가로/세로로 늘어나 보인다.

level별 유지 영역 (한 변 기준 / 면적 기준):
level 1: 95% / 약 90%
level 2: 90% / 81%
level 3: 80% / 64%
level 4: 70% / 49%

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 유지 비율 (한 변 기준)
LEVEL_RATIOS = {
    1: 0.95,
    2: 0.90,
    3: 0.80,
    4: 0.70,
}

# 원본 종횡비에 곱해지는 배율 범위 (1.0이면 원본과 같은 종횡비)
RATIO_RANGE = (3 / 4, 4 / 3)


def ResizeCrop(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    size: tuple = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in LEVEL_RATIOS:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # level이 정하는 면적 비율
    area_ratio = LEVEL_RATIOS[level] ** 2

    # 종횡비 배율을 랜덤으로 결정
    # (크롭이 이미지 밖으로 나가지 않는 범위로 제한)
    ratio_min = max(RATIO_RANGE[0], area_ratio)
    ratio_max = min(RATIO_RANGE[1], 1 / area_ratio)

    aspect_scale = math.exp(
        rng.uniform(
            math.log(ratio_min),
            math.log(ratio_max)
        )
    )

    crop_width = int(round(width * math.sqrt(area_ratio * aspect_scale)))
    crop_height = int(round(height * math.sqrt(area_ratio / aspect_scale)))

    crop_width = max(1, min(crop_width, width))
    crop_height = max(1, min(crop_height, height))

    # 크롭 위치를 랜덤으로 결정
    left = rng.randint(0, width - crop_width)
    top = rng.randint(0, height - crop_height)
    right = left + crop_width
    bottom = top + crop_height

    cropped = image.crop(
        (left, top, right, bottom)
    )

    if size is None:
        size = (width, height)

    resized = cropped.resize(
        size,
        Image.Resampling.LANCZOS
    )

    return resized