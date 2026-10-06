import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
GridDistort 변조 (랜덤 격자 왜곡)

이미지를 N x N 격자로 나눈 뒤, 격자선의 위치를 랜덤하게 옮겨
칸마다 서로 다른 비율로 늘어나거나 줄어들게 만든다.

가로 방향과 세로 방향을 각각 독립적으로 처리한다.
세로 격자선은 좌우로, 가로 격자선은 상하로 랜덤하게 이동하며,
격자선 사이의 내용은 그에 맞춰 늘어나거나 압축된다.
그래서 어떤 칸은 넓어지고 이웃 칸은 좁아지는 식의
불균일한 왜곡이 생긴다.

이미지의 바깥 테두리는 고정이라 내용이 잘리거나 여백이 생기지 않는다.
출력 크기는 원본과 같다.

level별 격자 수 / 격자선 최대 이동량 (이미지 가로·세로 대비):
level 1: 2x2 / 3%
level 2: 3x3 / 6%
level 3: 4x4 / 10%
level 4: 5x5 / 9%

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 격자 분할 수(N x N) 및 격자선 최대 이동 비율
CONFIGS = {
    1: {"grid_steps": 2, "distort_limit": 0.03},
    2: {"grid_steps": 3, "distort_limit": 0.06},
    3: {"grid_steps": 4, "distort_limit": 0.10},
    4: {"grid_steps": 5, "distort_limit": 0.09},
}


def axis_map(length, steps, limit, rng):
    # 한 축에 대해, 출력의 각 픽셀이 원본의 어느 위치를 참조할지 구한다.

    # 이웃한 격자선끼리 교차하지 않도록 이동량을 칸 크기의 45%로 제한
    max_shift = min(limit, 0.45 / steps) * length

    # 출력에서는 격자선이 균등하게 놓이고,
    # 원본에서는 안쪽 격자선이 랜덤하게 이동한 위치에 놓인다.
    dst_edges = np.linspace(0, length, steps + 1)
    src_edges = dst_edges.copy()

    for idx in range(1, steps):
        src_edges[idx] += rng.uniform(-max_shift, max_shift)

    # 격자선 사이는 선형 보간 (픽셀 중심 기준)
    coords = np.arange(length, dtype=np.float64) + 0.5

    return (
        np.interp(coords, dst_edges, src_edges) - 0.5
    ).astype(np.float32)


def GridDistort(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in CONFIGS:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    config = CONFIGS[level]
    steps = config["grid_steps"]
    limit = config["distort_limit"]

    width, height = image.size

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # 가로/세로 방향의 참조 위치를 각각 랜덤으로 생성
    map_x, map_y = np.meshgrid(
        axis_map(width, steps, limit, rng),
        axis_map(height, steps, limit, rng)
    )

    distorted = cv2.remap(
        np.asarray(image),
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101
    )

    GridDistort = Image.fromarray(distorted)

    return GridDistort