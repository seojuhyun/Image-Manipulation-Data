import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
OptDistort 변조 (랜덤 광학 렌즈 왜곡)

이미지 중심에서 멀어질수록 픽셀을 방사 방향으로 더 많이 밀거나 당겨,
렌즈나 굴절 매질을 통해 본 것처럼 휘어지게 만든다.
왜곡 계수의 크기는 level별 범위 안에서, 방향(부호)은 랜덤으로 정한다.

- 계수가 양수: 내용이 중심 쪽으로 모이고, 가장자리에는
  이미지를 거울처럼 반사한 내용이 채워진다.
- 계수가 음수: 가장자리가 바깥으로 늘어나 확대되어 보인다.
  계수가 크면 모서리 쪽에서 내용이 접혀 아치 모양이 생길 수 있다.

출력 크기는 원본과 같고, 검은 여백은 생기지 않는다.

level별 왜곡 계수 범위 (절댓값):
level 1: 0.05~0.15
level 2: 0.15~0.275
level 3: 0.275~0.425
level 4: 0.425~0.575

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 왜곡 계수 범위 (절댓값)
COEFF_RANGES = {
    1: (0.05, 0.15),
    2: (0.15, 0.275),
    3: (0.275, 0.425),
    4: (0.425, 0.575),
}


def OptDistort(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in COEFF_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # level별 범위 안에서 왜곡 계수를, 그리고 방향을 랜덤으로 결정
    k = rng.uniform(*COEFF_RANGES[level])

    if rng.random() < 0.5:
        k = -k

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    # 중심 기준으로 -1~1로 정규화한 출력 좌표
    xs = (np.arange(width, dtype=np.float32) - cx) / cx
    ys = (np.arange(height, dtype=np.float32) - cy) / cy
    nx, ny = np.meshgrid(xs, ys)

    # 중심에서 멀수록 커지는 방사형 왜곡
    factor = 1.0 + k * (nx * nx + ny * ny)

    # 출력의 각 픽셀이 원본의 어느 위치를 참조할지 계산
    map_x = (cx + nx * cx * factor).astype(np.float32)
    map_y = (cy + ny * cy * factor).astype(np.float32)

    # 이미지 밖을 참조하는 부분은 거울 반사로 채운다.
    distorted = cv2.remap(
        np.asarray(image),
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101
    )

    OptDistort = Image.fromarray(distorted)
    
    return OptDistort