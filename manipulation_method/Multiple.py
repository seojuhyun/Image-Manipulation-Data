import random
import zlib

from PIL import Image

"""
Multiple 변조 (랜덤 배율 곱하기)

랜덤으로 뽑은 배율 하나를 이미지의 모든 픽셀 값에 곱한다.
    출력 = 입력 x 배율   (255를 넘는 값은 255로 자른다)

배율이 1보다 크므로 이미지 전체가 밝아진다.
밝은 픽셀일수록 값이 많이 커지기 때문에, 밝은 영역부터 먼저 하얗게 날아가고
배율이 클수록 어두운 부분만 남고 나머지는 흰색으로 덮인다.
검은색(0)은 몇을 곱해도 0이라 변하지 않는다.

배율은 level별 범위 안에서 균등 분포로 뽑으며,
한 이미지 안에서는 모든 픽셀과 채널에 같은 배율을 쓴다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 배율 범위:
level 1: 1.1~1.3배
level 2: 1.3~1.7배
level 3: 1.7~2.5배
level 4: 2.5~4.0배

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 배율 범위
MULTIPLIER_RANGES = {
    1: (1.1, 1.3),
    2: (1.3, 1.7),
    3: (1.7, 2.5),
    4: (2.5, 4.0),
}

# True면 절반의 확률로 배율의 역수(1 / 배율)를 써서
# 밝게 하는 대신 어둡게 만든다.
ALLOW_DARKEN = False


def Multiple(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in MULTIPLIER_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # level별 범위 안에서 배율을 랜덤으로 결정
    multiplier = rng.uniform(*MULTIPLIER_RANGES[level])

    if ALLOW_DARKEN and rng.random() < 0.5:
        multiplier = 1.0 / multiplier

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    # 흑백(L)은 그대로, 그 외에는 RGB로 바꿔서 처리한다.
    base = image if image.mode == "L" else image.convert("RGB")

    # 0~255의 각 값에 배율을 곱한 결과를 미리 계산해 둔 표
    table = [
        min(255, round(value * multiplier))
        for value in range(256)
    ]

    # 표를 모든 채널에 적용한다.
    Multiple = base.point(table * len(base.getbands()))

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Multiple.putalpha(alpha)

    return Multiple