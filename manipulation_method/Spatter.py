import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
Spatter 변조 (랜덤 물방울 / 진흙 튀김)

카메라 렌즈나 유리창에 빗방울 또는 진흙이 튄 것처럼 얼룩을 덮는다.
(albumentations의 Spatter와 같은 방식)

1. 정규분포 난수로 된 "액체 층"을 만들고 가우시안으로 흐려 덩어리지게 한다.
2. 값이 큰 상위 몇 %(덮을 비율)만 방울로 남긴다.
3. 방식은 아래 두 가지 중에서 랜덤으로 고른다.
   - RAIN: 방울 테두리로부터의 거리 지도에 엠보스 필터를 적용해,
           빛이 맺힌 것처럼 입체적인 밝은 물방울을 더한다.
           방울 테두리는 살짝 어둡게 해 흰 배경 위에서도 방울이 보이게 한다.
   - MUD:  방울을 부드럽게 다듬은 뒤 진흙 색으로 덮는다. (탁한 갈색 얼룩)

방울 크기는 짧은 변 대비 비율로 정해 해상도가 달라도 비슷하게 보이게 한다.
방식, 방울 크기, 진흙 색, 액체 층은 이미지마다 랜덤으로 한 번 정해 lv1~lv4에 같이 쓰므로,
level이 오를수록 같은 자리의 방울이 커지고 새 방울이 늘어난다.
덮는 비율과 진하기는 level별 범위 안에서 랜덤으로 정한다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 덮는 비율 (이미지 넓이 대비) / 진하기:
level 1: 3~6%   / 0.45~0.60
level 2: 6~11%  / 0.55~0.70
level 3: 11~18% / 0.65~0.80
level 4: 18~28% / 0.75~0.90

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 튀김 설정
# coverage:  방울이 덮는 넓이 비율
# intensity: 물방울 하이라이트의 밝기 / 진흙의 불투명도
CONFIGS = {
    1: {"coverage": (0.03, 0.06), "intensity": (0.45, 0.60)},
    2: {"coverage": (0.06, 0.11), "intensity": (0.55, 0.70)},
    3: {"coverage": (0.11, 0.18), "intensity": (0.65, 0.80)},
    4: {"coverage": (0.18, 0.28), "intensity": (0.75, 0.90)},
}

# 방식 후보
MODES = ["RAIN", "MUD"]

# 액체 층을 흐리는 정도 = 방울 크기 (이미지 짧은 변 대비 / 720px 기준 약 2.9~5.8px)
DROP_SIZE_RANGE = (0.004, 0.008)

# 물방울 색 (밝은 하늘색 계열)
RAIN_COLOR = (200, 228, 240)

# 물방울 테두리를 어둡게 하는 정도
RAIN_RIM_SHADE = 0.35

# 진흙 색 후보
MUD_COLORS = [
    (74, 52, 34),
    (95, 78, 50),
    (60, 50, 42),
    (110, 95, 80),
]

# 진흙 얼룩에서 남길 가장자리 기준 (흐린 방울 지도의 값이 이보다 작으면 지운다)
MUD_CORE = 0.6

# True면 방식, 방울 크기, 진흙 색, 액체 층을 이미지마다 한 번만 골라 lv1~lv4에 같이 쓴다.
# False면 level마다 따로 고른다.
SAME_STYLE_ALL_LEVELS = True


def Spatter(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in CONFIGS:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # 너무 작은 이미지는 그대로 돌려준다.
    if width < 8 or height < 8:
        return image.copy()

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    # 방식을 고르는 style_rng는 이미지 내용만으로 시드를 만들어
    # lv1~lv4가 같은 방식을 쓰고 강도만 달라지게 한다.
    if rng is None:
        image_seed = zlib.crc32(image.tobytes())
        rng = random.Random(image_seed + level)
        style_rng = random.Random(image_seed) if SAME_STYLE_ALL_LEVELS else rng
    else:
        style_rng = rng

    config = CONFIGS[level]
    short_side = min(width, height)

    # 방식, 방울 크기, 진흙 색, 액체 층은 이미지마다 결정
    mode = style_rng.choice(MODES)
    drop_sigma = max(1.0, style_rng.uniform(*DROP_SIZE_RANGE) * short_side)
    mud_color = np.array(style_rng.choice(MUD_COLORS), dtype=np.float32) / 255.0
    field_rng = np.random.default_rng(style_rng.getrandbits(64))

    # 덮는 비율과 진하기는 level마다 결정
    coverage = rng.uniform(*config["coverage"])
    intensity = rng.uniform(*config["intensity"])

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB"))
    result = rgb.astype(np.float32) / 255.0

    # 1~2. 액체 층을 만들고 상위 coverage 비율만 방울로 남긴다.
    liquid = field_rng.standard_normal((height, width), dtype=np.float32)
    liquid = cv2.GaussianBlur(liquid, (0, 0), sigmaX=drop_sigma)
    threshold = np.quantile(liquid, 1.0 - coverage)
    drops = liquid > threshold

    if mode == "RAIN":
        drops_u8 = drops.astype(np.uint8) * 255

        # 방울 테두리로부터의 거리 지도
        edges = cv2.Canny(drops_u8, 50, 150)
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)

        truncate = max(3.0, 5.0 * drop_sigma)
        distance = np.minimum(distance, truncate)

        ksize = max(3, int(round(drop_sigma)) | 1)
        distance = cv2.blur(distance, (ksize, ksize))
        distance = cv2.equalizeHist(
            np.clip(distance / truncate * 255.0, 0, 255).astype(np.uint8)
        )

        # 엠보스 필터로 한쪽 테두리에 빛이 맺힌 듯한 입체감을 준다.
        emboss = np.array(
            [[-2, -1, 0],
             [-1, 1, 1],
             [0, 1, 2]],
            dtype=np.float32
        )
        highlight = cv2.filter2D(distance, -1, emboss)
        highlight = cv2.blur(highlight, (ksize, ksize)).astype(np.float32)

        highlight = drops.astype(np.float32) * highlight
        highlight /= max(float(highlight.max()), 1e-6)

        # 방울 테두리를 살짝 어둡게 한다.
        rim = cv2.GaussianBlur(
            edges.astype(np.float32) / 255.0, (0, 0), sigmaX=max(0.7, drop_sigma * 0.25)
        )
        rim /= max(float(rim.max()), 1e-6)

        rain_color = np.array(RAIN_COLOR, dtype=np.float32) / 255.0

        result = result * (1.0 - RAIN_RIM_SHADE * intensity * rim[..., None])
        result = result + highlight[..., None] * rain_color * intensity

    else:  # MUD
        mud = cv2.GaussianBlur(drops.astype(np.float32), (0, 0), sigmaX=drop_sigma)
        mud[mud < MUD_CORE] = 0.0
        mud = (mud * intensity)[..., None]

        result = result * (1.0 - mud) + mud_color * mud

    Spatter = Image.fromarray(
        np.clip(np.rint(result * 255.0), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Spatter.putalpha(alpha)

    return Spatter
