import math
import random
import zlib

from PIL import Image

"""
Skew 변조 (랜덤 기울이기)

이미지를 level별 각도 범위 안에서 랜덤한 각도만큼
가로 방향으로 기울여 평행사변형 형태로 만든다.
기울이는 방향(좌/우)도 랜덤이다.

이미지의 세로 중심을 기준으로 기울이며 캔버스 크기는 원본과 같다.
따라서 대각선 방향의 두 모서리에는 검은색 삼각형 여백이 생기고,
반대쪽 두 모서리의 내용은 캔버스 밖으로 나가 잘린다.
resize를 하지 않으므로 이미지가 눌리지는 않는다.

level별 기울기 각도 범위:
level 1: 1.25~3.75도
level 2: 3.75~7.5도
level 3: 7.5~12.5도
level 4: 12.5~17.5도

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 기울기 각도 범위 (도)
ANGLE_RANGES = {
    1: (1.25, 3.75),
    2: (3.75, 7.5),
    3: (7.5, 12.5),
    4: (12.5, 17.5),
}


def Skew(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in ANGLE_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # level별 범위 안에서 각도를, 그리고 방향을 랜덤으로 결정
    angle = rng.uniform(*ANGLE_RANGES[level])

    if rng.random() < 0.5:
        angle = -angle

    shear = math.tan(math.radians(angle))

    # 세로 중심을 기준으로 가로 방향 기울이기
    # PIL은 출력 좌표 -> 입력 좌표 방향의 계수를 요구한다.
    # x_in = x_out + shear * (y_out - height / 2)
    coeffs = (1, shear, -shear * height / 2, 0, 1, 0)

    skewed = image.transform(
        (width, height),
        Image.Transform.AFFINE,
        coeffs,
        Image.Resampling.BICUBIC
    )

    return skewed