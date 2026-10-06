import math
import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
WaterWave 변조 (랜덤 물결 효과)

이미지의 랜덤한 위치에 타원 모양의 물결(동심원 파문)을 하나 넣는다.
물에 돌을 던졌을 때처럼, 타원의 중심에서 바깥으로 퍼지는 고리 무늬가 생긴다.

타원 안쪽에는 두 가지를 적용한다.

- 굴절: 픽셀을 중심에서 바깥 방향(또는 안쪽 방향)으로 밀고 당기되,
  그 양을 중심으로부터의 거리에 따라 사인파로 바꾼다.
  그래서 고리마다 내용이 번갈아 늘어나고 줄어들며 일그러진다.
- 음영: 물결의 골에 해당하는 고리를 어둡게 만든다.
  하늘처럼 무늬가 없는 곳에서도 고리가 보이게 하기 위함이다.

타원 바깥쪽은 원본 그대로이고, 타원의 경계는 선명하게 끊긴다.
출력 크기는 원본과 같다.

물결의 위치, 타원의 가로/세로 비율, 고리 개수는 랜덤이고,
물결의 크기는 level별 범위 안에서 랜덤으로 정한다.
타원의 중심은 이미지 안에 놓이지만 일부가 이미지 밖으로 나갈 수 있다.

level별 물결 반지름 범위 (이미지 짧은 변 대비):
level 1: 15~20%
level 2: 20~27.5%
level 3: 27.5~35%
level 4: 35~45%

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 물결 반지름 범위 (이미지 짧은 변 대비)
RADIUS_RANGES = {
    1: (0.15, 0.20),
    2: (0.20, 0.275),
    3: (0.275, 0.35),
    4: (0.35, 0.45),
}

# 타원의 가로/세로 비율 범위 (1.0이면 원)
ASPECT_RANGE = (3 / 4, 4 / 3)

# 고리 개수 범위
RING_RANGE = (6, 14)

# 고리 간격 대비 미는 양의 범위 (클수록 내용이 심하게 일그러진다)
PUSH_RANGE = (0.15, 0.35)

# 물결의 골이 어두워지는 정도 (0이면 음영 없음, 1이면 골이 완전히 검게 됨)
SHADE_RANGE = (0.5, 0.8)


def WaterWave(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in RADIUS_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # 너무 작은 이미지에는 물결을 넣을 수 없으므로 그대로 돌려준다.
    if width < 2 or height < 2:
        return image.copy()

    # 팔레트(P), 흑백 1비트(1) 등은 픽셀값을 직접 섞을 수 없으므로 RGB로 바꾼다.
    if image.mode not in {"L", "LA", "RGB", "RGBA"}:
        image = image.convert("RGB")

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    short_side = min(width, height)

    # 물결의 크기와 타원 비율을 랜덤으로 결정
    radius = rng.uniform(*RADIUS_RANGES[level]) * short_side

    aspect = math.exp(
        rng.uniform(
            math.log(ASPECT_RANGE[0]),
            math.log(ASPECT_RANGE[1])
        )
    )

    radius_x = max(1.0, radius * math.sqrt(aspect))
    radius_y = max(1.0, radius / math.sqrt(aspect))

    # 물결의 중심 위치를 랜덤으로 결정
    center_x = rng.uniform(0, width - 1)
    center_y = rng.uniform(0, height - 1)

    # 고리 개수, 미는 양, 음영의 세기, 물결의 시작 위상을 랜덤으로 결정
    rings = rng.randint(*RING_RANGE)
    push = rng.uniform(*PUSH_RANGE) / rings
    shade = rng.uniform(*SHADE_RANGE)
    phase = rng.uniform(0, 2 * math.pi)

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32)
    )

    # 타원 중심 기준 좌표를 타원 반지름으로 나눠 정규화한다.
    # (dist가 1이면 타원의 경계)
    norm_x = (grid_x - center_x) / radius_x
    norm_y = (grid_y - center_y) / radius_y
    dist = np.sqrt(norm_x * norm_x + norm_y * norm_y)

    inside = dist < 1.0

    # 중심으로부터의 거리에 따른 물결의 위상
    wave = 2 * np.pi * rings * dist + phase

    # 거리에 따라 사인파 모양으로 밀고 당긴다.
    offset = push * np.sin(wave)

    # 미는 방향은 중심에서 바깥으로 향하는 방향이다.
    safe_dist = np.maximum(dist, 1e-6)
    shift_x = offset * (norm_x / safe_dist) * radius_x
    shift_y = offset * (norm_y / safe_dist) * radius_y

    # 타원 바깥쪽은 원본 그대로 둔다.
    map_x = np.where(inside, grid_x + shift_x, grid_x).astype(np.float32)
    map_y = np.where(inside, grid_y + shift_y, grid_y).astype(np.float32)

    # 이미지 밖을 참조하는 부분은 거울 반사로 채운다.
    distorted = cv2.remap(
        np.asarray(image),
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # 물결의 골을 어둡게 만든다 (타원 바깥쪽은 밝기 1 그대로).
    trough = (0.5 + 0.5 * np.cos(wave)) ** 2
    brightness = np.where(inside, 1.0 - shade * trough, 1.0)

    # 투명도(알파) 채널은 어둡게 하지 않는다.
    shaded = distorted.astype(np.float32)

    if shaded.ndim == 2:
        shaded = shaded * brightness
    else:
        color_channels = 3 if shaded.shape[2] >= 3 else 1
        shaded[..., :color_channels] *= brightness[..., None]

    WaterWave = Image.fromarray(
        np.clip(np.rint(shaded), 0, 255).astype(np.uint8)
    )

    return WaterWave