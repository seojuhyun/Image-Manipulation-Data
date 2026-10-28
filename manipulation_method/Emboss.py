import math
import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
Emboss 변조 (랜덤 엠보싱)

빛이 한쪽에서 비스듬히 비추는 것처럼, 경계의 한쪽에는 밝은 하이라이트를,
반대쪽에는 어두운 그림자를 넣어 그림이 종이에 눌려 튀어나온 듯한
3D 부조(relief) 느낌을 만든다.

1. 이미지를 흑백 밝기(0~1)로 바꾼다.
2. 빛의 방향으로 조금 옮긴 사본과 반대 방향으로 옮긴 사본의 차이를 구한다.
   (= 빛 방향의 밝기 변화량, 경계에서만 값이 생긴다. 범위 -1~1)
3. 이 값을 depth배 해서
   - 컬러 엠보스: 원본의 R, G, B에 똑같이 더해 색은 유지한 채 입체감만 준다.
   - 회색 엠보스: 회색(128) 바탕에 더해 전형적인 엠보스 필터 결과를 만든다.
4. 두 결과를 회색 비율(gray)만큼 섞는다.

level이 오를수록 입체감(depth)이 강해지고, 색이 빠지며 회색 엠보스에 가까워진다.
level 4는 거의 회색 바탕에 윤곽만 도드라진 전형적인 엠보스 이미지가 된다.

빛의 방향(0~360도)과 경계 두께는 이미지마다 랜덤으로 한 번 정해 lv1~lv4에 같이 쓴다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 입체감(depth) / 회색 엠보스 비율(gray) 범위:
level 1: 0.5~0.8 / 0%
level 2: 0.8~1.2 / 15~30%
level 3: 1.2~1.8 / 45~65%
level 4: 1.8~2.6 / 85~100%

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 엠보스 설정
# depth: 하이라이트/그림자의 세기
# gray:  회색 엠보스 결과를 섞는 비율 (0이면 색 유지, 1이면 완전한 회색 엠보스)
CONFIGS = {
    1: {"depth": (0.5, 0.8), "gray": (0.00, 0.00)},
    2: {"depth": (0.8, 1.2), "gray": (0.15, 0.30)},
    3: {"depth": (1.2, 1.8), "gray": (0.45, 0.65)},
    4: {"depth": (1.8, 2.6), "gray": (0.85, 1.00)},
}

# 사본을 옮기는 거리 = 하이라이트/그림자 띠의 두께
# (이미지 짧은 변 대비 / 짧은 변 720px 기준 약 1.1~2.2px)
BEVEL_RANGE = (0.0015, 0.003)

# True면 빛의 방향과 경계 두께을(를) 이미지마다 한 번만 골라 lv1~lv4에 같이 쓴다. (강도만 level별로 다름)
# False면 level마다 따로 고른다.
SAME_STYLE_ALL_LEVELS = True


def shift(gray, dx, dy):
    # 이미지를 (dx, dy)만큼 옮긴다. (소수점 이동 가능, 바깥은 가장자리 값으로 채움)
    matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(
        gray,
        matrix,
        (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


def Emboss(
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
    if width < 3 or height < 3:
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

    # 빛의 방향과 경계 두께는 이미지마다, 입체감과 회색 비율은 level마다 랜덤으로 결정
    angle = style_rng.uniform(0, 2 * math.pi)
    bevel = max(1.0, style_rng.uniform(*BEVEL_RANGE) * min(width, height))
    depth = rng.uniform(*config["depth"])
    gray_mix = rng.uniform(*config["gray"])

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) / 255.0

    # 계단 현상을 막기 위해 아주 살짝 흐린 뒤 계산한다.
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=bevel * 0.5)

    # 빛 방향으로 옮긴 사본과 반대로 옮긴 사본의 차이 (-1 ~ 1)
    dx = math.cos(angle) * bevel
    dy = math.sin(angle) * bevel
    relief = shift(gray, dx, dy) - shift(gray, -dx, -dy)

    offset = relief * depth * 127.5

    # 컬러 엠보스: 원본 색에 하이라이트/그림자를 더한다.
    colored = rgb + offset[..., None]

    # 회색 엠보스: 회색 바탕에 하이라이트/그림자를 더한다.
    embossed_gray = 127.5 + offset

    mixed = (1.0 - gray_mix) * colored + gray_mix * embossed_gray[..., None]

    Emboss = Image.fromarray(
        np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Emboss.putalpha(alpha)

    return Emboss
