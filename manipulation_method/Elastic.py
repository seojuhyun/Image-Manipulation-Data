import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
Elastic 변조 (랜덤 탄성 왜곡)

이미지를 젤리 덩어리처럼 통째로 비틀고 출렁이게 만든다.
두 가지 변형을 함께 적용한다.

- 전체 변형: 이미지 가운데에 놓은 정사각형의 세 꼭짓점을 각각 랜덤한
  방향으로 옮기고, 그에 맞춰 이미지 전체를 따라 움직인다(아핀 변환).
  그 결과 이미지가 회전하고, 기울어지고, 늘어나거나 줄어든다.
- 국소 변형: 성긴 격자의 격자점마다 랜덤한 이동량을 정하고 이를 부드럽게
  이어, 위치마다 조금씩 다르게 출렁이도록 만든다.

변형으로 생긴 빈 영역을 채우는 방식은 랜덤으로 고른다.
- 검은색으로 채우기
- 이미지를 거울처럼 반사해 채우기

출력 크기는 원본과 같다.

level별 전체 변형 / 국소 변형의 최대 크기:
level 1: 꼭짓점 이동 20% / 출렁임 1%
level 2: 꼭짓점 이동 40% / 출렁임 2%
level 3: 꼭짓점 이동 70% / 출렁임 3%
level 4: 꼭짓점 이동 100% / 출렁임 4%

꼭짓점 이동은 기준 정사각형의 한 변의 절반(이미지 짧은 변의 1/3) 대비,
출렁임은 이미지 짧은 변 대비 비율이다.

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 전체 변형(꼭짓점 이동) / 국소 변형(출렁임)의 최대 크기
CONFIGS = {
    1: {"affine": 0.20, "wobble": 0.01},
    2: {"affine": 0.40, "wobble": 0.02},
    3: {"affine": 0.70, "wobble": 0.03},
    4: {"affine": 1.00, "wobble": 0.04},
}

# 국소 변형에서 짧은 변을 몇 칸으로 나눌지 (클수록 출렁임이 잘게 생긴다)
GRID_CELLS = 4

# 전체 변형으로 면적이 바뀌는 배율의 허용 범위
# (이 범위를 벗어나면 이미지가 뒤집히거나 납작해지므로 다시 뽑는다)
AREA_SCALE_RANGE = (0.4, 2.5)


def Elastic(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in CONFIGS:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # 너무 작은 이미지는 변형할 수 없으므로 그대로 돌려준다.
    if width < 2 or height < 2:
        return image.copy()

    # 팔레트(P), 흑백 1비트(1) 등은 픽셀값을 직접 섞을 수 없으므로 RGB로 바꾼다.
    if image.mode not in {"L", "LA", "RGB", "RGBA"}:
        image = image.convert("RGB")

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    config = CONFIGS[level]
    short_side = min(width, height)

    # ------------------------------------------------------------
    # 전체 변형: 가운데 정사각형의 세 꼭짓점을 랜덤하게 옮긴다.
    # ------------------------------------------------------------

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    half = short_side / 3.0

    # 기준 정사각형의 세 꼭짓점 (우하, 우상, 좌상)
    src_points = np.array(
        [
            (cx + half, cy + half),
            (cx + half, cy - half),
            (cx - half, cy - half),
        ],
        dtype=np.float32
    )

    max_shift = config["affine"] * half

    # 최댓값의 50~100% 크기로, 방향(부호)은 랜덤으로 옮긴다.
    def shift():
        value = rng.uniform(0.5, 1.0) * max_shift
        return value if rng.random() < 0.5 else -value

    # 이미지가 뒤집히거나 납작해지는 변환이 나오면 다시 뽑는다.
    for _ in range(50):
        dst_points = src_points + np.array(
            [(shift(), shift()) for _ in range(3)],
            dtype=np.float32
        )

        # 출력 좌표 -> 입력 좌표 방향의 아핀 변환 행렬 (2 x 3)
        matrix = cv2.getAffineTransform(dst_points, src_points)

        area_scale = 1.0 / max(
            1e-12,
            matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
        )

        if AREA_SCALE_RANGE[0] <= area_scale <= AREA_SCALE_RANGE[1]:
            break
    else:
        # 끝내 적당한 변환이 나오지 않으면 전체 변형은 생략한다.
        matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

    # ------------------------------------------------------------
    # 국소 변형: 격자점마다 랜덤한 이동량을 정해 부드럽게 잇는다.
    # ------------------------------------------------------------

    amplitude = config["wobble"] * short_side

    # 격자 칸이 정사각형에 가깝도록 가로/세로 칸 수를 정한다.
    cell_size = short_side / GRID_CELLS
    cols = max(1, round(width / cell_size))
    rows = max(1, round(height / cell_size))

    def random_grid():
        return np.array(
            [
                [rng.uniform(-1, 1) for _ in range(cols + 1)]
                for _ in range(rows + 1)
            ],
            dtype=np.float32
        )

    dx = cv2.resize(
        random_grid(), (width, height), interpolation=cv2.INTER_CUBIC
    ) * amplitude

    dy = cv2.resize(
        random_grid(), (width, height), interpolation=cv2.INTER_CUBIC
    ) * amplitude

    # ------------------------------------------------------------
    # 두 변형을 합쳐, 출력의 각 픽셀이 원본의 어느 위치를 참조할지 계산
    # ------------------------------------------------------------

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32)
    )

    # 먼저 국소 변형만큼 옮긴 뒤, 전체 변형을 적용한다.
    moved_x = grid_x + dx
    moved_y = grid_y + dy

    map_x = (
        matrix[0, 0] * moved_x + matrix[0, 1] * moved_y + matrix[0, 2]
    ).astype(np.float32)

    map_y = (
        matrix[1, 0] * moved_x + matrix[1, 1] * moved_y + matrix[1, 2]
    ).astype(np.float32)

    # 빈 영역을 채우는 방식을 랜덤으로 결정 (검은색 / 거울 반사)
    if rng.random() < 0.5:
        border_mode = cv2.BORDER_CONSTANT
    else:
        border_mode = cv2.BORDER_REFLECT_101

    distorted = cv2.remap(
        np.asarray(image),
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=border_mode,
        borderValue=0
    )

    Elastic = Image.fromarray(distorted)

    return Elastic