import random
import zlib

from PIL import Image

"""
Gamma 변조 (랜덤 감마 변환)

각 픽셀 값을 0~1 범위로 바꾼 뒤 gamma 제곱을 해서 밝기를 바꾼다.
    출력 = 입력 ^ gamma

gamma가 1보다 크면 중간 밝기가 어두워지고, 클수록 더 어두워진다.
완전한 검은색(0)과 흰색(1)은 변하지 않고 그 사이의 값만 바뀌므로,
어두운 부분은 검게 뭉개지고 대비가 강해진다.
R, G, B 채널에 각각 적용하기 때문에 색도 더 짙어진다.

gamma 값은 level별 범위 안에서 랜덤으로 정한다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 gamma 범위:
level 1: 1.2~1.5
level 2: 1.5~2.0
level 3: 2.0~3.0
level 4: 3.0~4.0

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 gamma 범위
GAMMA_RANGES = {
    1: (1.2, 1.5),
    2: (1.5, 2.0),
    3: (2.0, 3.0),
    4: (3.0, 4.0),
}

# True면 절반의 확률로 gamma의 역수(1 / gamma)를 써서
# 어둡게 하는 대신 밝게 만든다.
ALLOW_BRIGHTEN = False


def Gamma(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in GAMMA_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # level별 범위 안에서 gamma 값을 랜덤으로 결정
    gamma = rng.uniform(*GAMMA_RANGES[level])

    if ALLOW_BRIGHTEN and rng.random() < 0.5:
        gamma = 1.0 / gamma

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    # 흑백(L)은 그대로, 그 외에는 RGB로 바꿔서 처리한다.
    base = image if image.mode == "L" else image.convert("RGB")

    # 0~255의 각 값이 변환 후 어떤 값이 되는지 미리 계산해 둔 표
    # (출력 = 255 x (입력 / 255) ^ gamma)
    table = [
        round(255.0 * (value / 255.0) ** gamma)
        for value in range(256)
    ]

    # 표를 모든 채널에 적용한다.
    Gamma = base.point(table * len(base.getbands()))

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Gamma.putalpha(alpha)

    return Gamma