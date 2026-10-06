import random
import zlib

import numpy as np
from PIL import Image

"""
Zooming 변조 (랜덤 줌 블러)

촬영 중에 줌을 움직였을 때 생기는 방사형 흐림을 흉내 낸다.
이미지 중심을 기준으로 배율을 1.0배부터 최대 배율까지 조금씩 키워 가며
확대한 이미지를 여러 장 만들고, 이를 모두 평균한다.
그 결과 중심은 비교적 선명하고, 가장자리로 갈수록
중심에서 바깥 방향으로 번지듯 흐려진다.

최대 배율은 level별 범위 안에서 랜덤으로 정하며,
배율이 클수록 흐림이 강하다.
노출 중 줌 인과 줌 아웃은 같은 배율 구간을 지나므로 결과가 같아
방향은 따로 구분하지 않는다.
출력 크기는 원본과 같다.

level별 최대 줌 배율 범위:
level 1: 1.03~1.08배
level 2: 1.08~1.15배
level 3: 1.15~1.25배
level 4: 1.25~1.35배

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 최대 줌 배율 범위
ZOOM_RANGES = {
    1: (1.03, 1.08),
    2: (1.08, 1.15),
    3: (1.15, 1.25),
    4: (1.25, 1.35),
}

# 평균할 이미지들 사이의 배율 간격
ZOOM_STEP = 0.01


def Zooming(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in ZOOM_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # level별 범위 안에서 최대 줌 배율을 랜덤으로 결정
    max_zoom = rng.uniform(*ZOOM_RANGES[level])

    # 1.0배(원본)부터 최대 배율 직전까지의 배율 목록
    zoom_factors = np.arange(1.0, max_zoom, ZOOM_STEP)

    accumulated = np.zeros_like(
        np.asarray(image),
        dtype=np.float32
    )

    for factor in zoom_factors:

        # 중심 기준으로 1/factor 크기 영역을 잘라 원래 크기로 확대
        crop_width = width / factor
        crop_height = height / factor

        left = (width - crop_width) / 2
        top = (height - crop_height) / 2

        frame = image.resize(
            (width, height),
            Image.Resampling.BILINEAR,
            box=(left, top, left + crop_width, top + crop_height)
        )

        accumulated += np.asarray(frame, dtype=np.float32)

    # 모든 배율의 이미지를 평균
    blurred = accumulated / len(zoom_factors)

    Zooming = Image.fromarray(
        np.clip(np.rint(blurred), 0, 255).astype(np.uint8)
    )

    return Zooming