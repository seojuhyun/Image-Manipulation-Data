import math
import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
Swirl 변조 (랜덤 소용돌이)

이미지의 랜덤한 위치를 중심으로 내용을 소용돌이 모양으로 감아 돌린다.

각 픽셀을 소용돌이 중심 둘레로 회전시키되, 회전각을 중심에서 가장 크게 하고
중심에서 멀어질수록 지수적으로 줄인다.
그래서 중심 부근은 여러 바퀴 감기고, 바깥쪽은 완만하게 휘어진다.
(scikit-image의 swirl 변환과 같은 방식이다.)

소용돌이의 중심 위치, 회전 방향, 영향 범위는 랜덤이고,
중심에서의 회전각은 level별 범위 안에서 랜덤으로 정한다.
이미지 밖을 참조하는 부분은 거울 반사로 채워 검은 여백이 생기지 않는다.
출력 크기는 원본과 같다.

level별 중심에서의 회전각 범위 (라디안 / 바퀴 수):
level 1: 1.5~3   / 약 0.25~0.5바퀴
level 2: 3~6     / 약 0.5~1바퀴
level 3: 6~12    / 약 1~2바퀴
level 4: 12~20   / 약 2~3바퀴

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 중심에서의 회전각 범위 (라디안)
STRENGTH_RANGES = {
    1: (1.5, 3.0),
    2: (3.0, 6.0),
    3: (6.0, 12.0),
    4: (12.0, 20.0),
}

# 소용돌이의 영향 범위 (이미지 짧은 변 대비 반지름)
# 이 반지름의 약 1/7 거리마다 회전각이 절반으로 줄어든다.
RADIUS_RANGE = (2.0, 4.0)

# 소용돌이 중심이 놓일 수 있는 범위 (이미지 가로/세로 대비)
CENTER_RANGE = (0.2, 0.8)


def Swirl(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in STRENGTH_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # 너무 작은 이미지는 소용돌이를 만들 수 없으므로 그대로 돌려준다.
    if width < 2 or height < 2:
        return image.copy()

    # 팔레트(P), 흑백 1비트(1) 등은 픽셀값을 직접 섞을 수 없으므로 RGB로 바꾼다.
    if image.mode not in {"L", "LA", "RGB", "RGBA"}:
        image = image.convert("RGB")

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # 소용돌이의 중심 위치를 랜덤으로 결정
    center_x = rng.uniform(*CENTER_RANGE) * (width - 1)
    center_y = rng.uniform(*CENTER_RANGE) * (height - 1)

    # 중심에서의 회전각과 회전 방향을 랜덤으로 결정
    strength = rng.uniform(*STRENGTH_RANGES[level])

    if rng.random() < 0.5:
        strength = -strength

    # 소용돌이의 영향 범위를 랜덤으로 결정
    radius = rng.uniform(*RADIUS_RANGE) * min(width, height)

    # 회전각이 1/e로 줄어드는 거리
    decay = radius * math.log(2) / 5.0

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32) - center_x,
        np.arange(height, dtype=np.float32) - center_y
    )

    # 소용돌이 중심 기준 극좌표
    rho = np.sqrt(grid_x * grid_x + grid_y * grid_y)
    theta = np.arctan2(grid_y, grid_x)

    # 중심에 가까울수록 더 많이 회전시킨다.
    theta = theta + strength * np.exp(-rho / decay)

    # 출력의 각 픽셀이 원본의 어느 위치를 참조할지 계산
    map_x = (center_x + rho * np.cos(theta)).astype(np.float32)
    map_y = (center_y + rho * np.sin(theta)).astype(np.float32)

    # 이미지 밖을 참조하는 부분은 거울 반사로 채운다.
    distorted = cv2.remap(
        np.asarray(image),
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101
    )

    Swirl = Image.fromarray(distorted)

    return Swirl