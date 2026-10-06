import random
import zlib

from PIL import Image, ImageOps

"""
Posterize 변조 (랜덤 비트 수 감소)

각 픽셀의 R, G, B 값은 원래 8비트(0~255, 256단계)로 표현된다.
이 중 위쪽 몇 비트만 남기고 아래쪽 비트를 모두 0으로 지워,
채널마다 쓸 수 있는 밝기 단계의 수를 줄인다.

    남기는 비트 수   채널당 단계 수   표현 가능한 색의 수
         7               128            약 210만
         4                16            4,096
         2                 4            64
         1                 2            8

색이 부드럽게 변하던 곳(하늘, 그러데이션)이 계단처럼 끊어지고,
비트 수가 적을수록 포스터처럼 몇 가지 색으로만 칠한 모습이 된다.
아래쪽 비트를 버리기만 하므로 값은 항상 같거나 작아져,
전체적으로 조금 어두워진다. (예: 1비트만 남기면 255는 128이 된다.)

남기는 비트 수는 level별 후보 중에서 랜덤으로 고른다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 남기는 비트 수:
level 1: 6 또는 7비트
level 2: 4 또는 5비트
level 3: 2 또는 3비트
level 4: 1비트

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 남기는 비트 수 후보
BIT_CHOICES = {
    1: [6, 7],
    2: [4, 5],
    3: [2, 3],
    4: [1],
}


def Posterize(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in BIT_CHOICES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # level별 후보 중에서 남길 비트 수를 랜덤으로 선택
    bits = rng.choice(BIT_CHOICES[level])

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    # 흑백(L)은 그대로, 그 외에는 RGB로 바꿔서 처리한다.
    base = image if image.mode == "L" else image.convert("RGB")

    # 각 채널에서 위쪽 bits개 비트만 남긴다.
    Posterize = ImageOps.posterize(base, bits)

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Posterize.putalpha(alpha)

    return Posterize