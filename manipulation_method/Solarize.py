import random
import zlib

from PIL import Image, ImageOps

"""
Solarize 변조 (랜덤 임계값 반전)

밝기가 임계값 이상인 픽셀 값만 뒤집는다. (값 v를 255 - v로)
임계값보다 어두운 픽셀은 그대로 두므로, 밝은 영역만 어둡게 뒤집혀
필름을 과다 노출시킨 것 같은 강한 대비가 생긴다.
R, G, B 채널마다 따로 판단하기 때문에 색도 함께 바뀐다.

임계값은 level별 범위 안에서 랜덤으로 정한다.
임계값이 낮을수록 더 많은 픽셀이 뒤집혀 효과가 강하다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 임계값 범위:
level 1: 192~240 (아주 밝은 픽셀만 반전)
level 2: 128~192
level 3: 64~128
level 4: 1~64   (거의 모든 픽셀을 반전)

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 임계값 범위
THRESHOLD_RANGES = {
    1: (192, 240),
    2: (128, 192),
    3: (64, 128),
    4: (1, 64),
}


def Solarize(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in THRESHOLD_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # level별 범위 안에서 임계값을 랜덤으로 결정
    threshold = rng.randint(*THRESHOLD_RANGES[level])

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    # 흑백(L)은 그대로, 그 외에는 RGB로 바꿔서 처리한다.
    base = image if image.mode == "L" else image.convert("RGB")

    # 임계값 이상인 픽셀 값만 뒤집는다.
    Solarize = ImageOps.solarize(base, threshold=threshold)

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Solarize.putalpha(alpha)

    return Solarize