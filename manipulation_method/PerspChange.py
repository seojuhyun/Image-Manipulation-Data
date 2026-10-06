import random
import zlib

import numpy as np
from PIL import Image

"""
PerspChange 변조 (랜덤 원근 변환)

이미지의 네 모서리를 각각 독립적으로, 랜덤한 양만큼 안쪽으로 이동시켜
비대칭 사다리꼴 형태의 원근 왜곡을 만든다.
출력 크기는 원본과 같고, 이미지가 줄어들며 생긴 빈 영역은
검은색으로 채워진다.

이동량의 최댓값은 level로 정해지고, 실제 이동량은
모서리마다 0~최댓값 사이에서 랜덤으로 뽑는다.

level별 모서리 최대 이동 비율 (이미지 가로/세로 대비):
level 1: 2.5%
level 2: 5%
level 3: 10%
level 4: 15%

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 왜곡 계수 (이미지 크기 대비 모서리 최대 이동 비율)
DISTORTION_FACTORS = {
    1: 0.025,
    2: 0.05,
    3: 0.10,
    4: 0.15,
}


def find_coeffs(pa, pb):
    # pa의 네 점을 pb의 네 점으로 보내는 원근 변환 계수 8개를 구한다.

    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.array(matrix, dtype=float)
    B = np.array(pb, dtype=float).reshape(8)

    return np.linalg.solve(A, B)


def PerspChange(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in DISTORTION_FACTORS:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    factor = DISTORTION_FACTORS[level]
    width, height = image.size

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    max_dx = width * factor
    max_dy = height * factor

    # 원본 이미지의 네 꼭짓점 (좌상, 좌하, 우하, 우상)
    start_points = [
        (0, 0),
        (0, height),
        (width, height),
        (width, 0),
    ]

    # 각 모서리를 독립적으로 0~max만큼 안쪽으로 랜덤 이동
    def jitter():
        return rng.uniform(0, max_dx), rng.uniform(0, max_dy)

    j_tl, j_bl, j_br, j_tr = jitter(), jitter(), jitter(), jitter()

    end_points = [
        (j_tl[0], j_tl[1]),                          # 좌상
        (j_bl[0], height - j_bl[1]),                 # 좌하
        (width - j_br[0], height - j_br[1]),         # 우하
        (width - j_tr[0], j_tr[1]),                  # 우상
    ]

    # PIL은 출력 좌표 -> 입력 좌표 방향의 계수를 요구한다.
    coeffs = find_coeffs(end_points, start_points)

    transformed = image.transform(
        (width, height),
        Image.Transform.PERSPECTIVE,
        coeffs,
        Image.Resampling.BICUBIC,
    )

    return transformed