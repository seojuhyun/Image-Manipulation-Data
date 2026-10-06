import math
import random
import zlib

from PIL import Image, ImageOps

"""
Padding 변조 (랜덤 색상 패딩 + 리사이즈)

이미지 외곽에 랜덤한 단색 여백을 추가한 뒤
원래 이미지 크기로 resize한다.
좌우 패딩끼리, 상하 패딩끼리는 크기가 같아 이미지는 항상 중앙에 놓인다.

결과에서 원본이 차지하는 면적은 level로 고정이고,
패딩을 가로/세로에 어떻게 나눌지와 패딩 색은 랜덤이다.
가로/세로 패딩 양이 다른 만큼 resize 과정에서
이미지가 가로 또는 세로로 눌려 보인다.

level별 패딩 강도 (상하좌우 균등 패딩 기준 / 원본이 차지하는 면적):
level 1: 5% / 약 83%
level 2: 10% / 약 69%
level 3: 15% / 약 59%
level 4: 20% / 약 51%

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 패딩 비율 (상하좌우 균등 패딩 기준)
PADDING_RATIOS = {
    1: 0.05,
    2: 0.10,
    3: 0.15,
    4: 0.20,
}

# 원본이 차지하는 영역의 가로/세로 비율 변화 범위
# (1.0이면 가로·세로 패딩이 균등)
RATIO_RANGE = (1 / 2, 2)


def Padding(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in PADDING_RATIOS:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # level이 정하는, 결과에서 원본이 차지하는 면적 비율
    ratio = PADDING_RATIOS[level]
    area_ratio = (1 / (1 + 2 * ratio)) ** 2

    # 그 면적을 가로/세로에 어떻게 나눌지 랜덤으로 결정
    # (패딩이 음수가 되지 않는 범위로 제한)
    ratio_min = max(RATIO_RANGE[0], area_ratio)
    ratio_max = min(RATIO_RANGE[1], 1 / area_ratio)

    aspect_scale = math.exp(
        rng.uniform(
            math.log(ratio_min),
            math.log(ratio_max)
        )
    )

    # 결과에서 원본이 차지하는 가로/세로 비율
    content_w = math.sqrt(area_ratio * aspect_scale)
    content_h = math.sqrt(area_ratio / aspect_scale)

    # 한쪽 패딩 픽셀 크기 계산
    pad_w = int(round(width * (1 / content_w - 1) / 2))
    pad_h = int(round(height * (1 / content_h - 1) / 2))

    # 패딩 색을 랜덤으로 결정
    color = (
        rng.randint(0, 255),
        rng.randint(0, 255),
        rng.randint(0, 255),
    )

    # 외곽 여백 추가 (left, top, right, bottom)
    padded = ImageOps.expand(
        image,
        border=(pad_w, pad_h, pad_w, pad_h),
        fill=color
    )

    # 원래 이미지 크기로 다시 리사이즈
    padding = padded.resize(
        (width, height),
        Image.Resampling.LANCZOS
    )

    return padding