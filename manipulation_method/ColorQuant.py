import random
import zlib

from PIL import Image

"""
ColorQuant 변조 (랜덤 색상 수 감소)

이미지에 쓰인 색의 가짓수를 정해진 개수 이하로 줄인다.

1. 이미지 전체의 색 분포를 보고, 대표색을 정해진 개수만큼 뽑아
   팔레트를 만든다.
2. 모든 픽셀을 팔레트에서 가장 가까운 색으로 바꾼다.

색이 부드럽게 변하던 곳(하늘, 그러데이션)은 몇 가지 색의 띠로 끊어지고,
색 수가 적을수록 포스터처럼 평평하게 칠한 모습이 된다.
형태와 위치는 그대로이다.

남길 색의 개수는 level별 범위 안에서 랜덤으로 정하고,
대표색을 뽑는 방법도 아래 세 가지 중에서 랜덤으로 고른다.
- MEDIANCUT:   픽셀 수가 고르게 나뉘도록 색 공간을 쪼갠다. 원본에 가장 가깝다.
- MAXCOVERAGE: 색 공간을 넓게 덮도록 뽑는다. 원본과 다른 색이 섞이기 쉽다.
- FASTOCTREE:  색 공간을 8등분씩 나눠 가며 뽑는다. 흐린 색이 뭉개지기 쉽다.

출력은 RGB이고(투명도가 있는 이미지는 RGBA), 출력 크기는 원본과 같다.

level별 색상 수 범위:
level 1: 64~128개
level 2: 32~64개
level 3: 16~32개
level 4: 4~16개

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 색상 수 범위
COLOR_RANGES = {
    1: (64, 128),
    2: (32, 64),
    3: (16, 32),
    4: (4, 16),
}

# 대표색을 뽑는 방법 후보
QUANTIZE_METHODS = [
    Image.Quantize.MEDIANCUT,
    Image.Quantize.MAXCOVERAGE,
    Image.Quantize.FASTOCTREE,
]


def ColorQuant(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in COLOR_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # 남길 색상 수와 대표색을 뽑는 방법을 랜덤으로 결정
    num_colors = rng.randint(*COLOR_RANGES[level])
    method = rng.choice(QUANTIZE_METHODS)

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    # 팔레트를 만들고 모든 픽셀을 팔레트의 색으로 바꾼다.
    quantized = image.convert("RGB").quantize(
        colors=num_colors,
        method=method
    )

    # 팔레트 이미지를 다시 RGB로 되돌린다.
    ColorQuant = quantized.convert("RGB")

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        ColorQuant.putalpha(alpha)

    return ColorQuant