import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
Dehaze 변조 (Dark Channel Prior 안개 제거)

He et al.(CVPR 2009)의 Dark Channel Prior로 이미지에서 "안개"를 걷어낸다.
안개가 없는 이미지에 적용하면, 밝고 흐린 색(파스텔 톤)을 안개로 보고 걷어내므로
색이 진해지고 어두워지며 대비가 강해진다.

1. 다크 채널: 픽셀마다 R, G, B 중 가장 작은 값을 고르고, 주변 패치 안의 최솟값을 구한다.
   선명한 영역은 이 값이 0에 가깝고, 뿌옇게 밝은 영역은 크다.
2. 대기광(A): 다크 채널이 가장 밝은 상위 0.1% 픽셀들의 평균 색. (보통 흰 배경, 하늘)
3. 투과율: t = 1 - omega x (A로 나눈 이미지의 다크 채널)
   omega는 걷어낼 안개의 비율이다. 클수록 더 많이 걷어낸다.
4. 투과율 지도를 원본 밝기를 길잡이로 부드럽게 다듬는다. (guided filter)
5. 복원: J = (I - A) / max(t, T_MIN) + A

웹툰처럼 흰 배경이 많은 이미지는 흰색이 대기광이 되어 흰 배경은 그대로 남고,
연한 색 영역일수록 진하고 어둡게 바뀐다. 이미 어두운 영역(검은 선)은 거의 변하지 않는다.

omega는 level별 범위 안에서 랜덤으로 정한다.
출력은 RGB이고(투명도가 있는 이미지는 RGBA), 출력 크기는 원본과 같다.

level별 omega 범위:
level 1: 0.35~0.50
level 2: 0.50~0.65
level 3: 0.65~0.80
level 4: 0.80~0.95 (원 논문 기본값 0.95)

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 걷어낼 안개의 비율(omega) 범위
OMEGA_RANGES = {
    1: (0.35, 0.50),
    2: (0.50, 0.65),
    3: (0.65, 0.80),
    4: (0.80, 0.95),
}

# 다크 채널을 구하는 패치 크기 (이미지 짧은 변 대비 / 720px 기준 약 15px)
PATCH_RATIO = 0.02

# guided filter 반경 (이미지 짧은 변 대비 / 720px 기준 약 58px)
GUIDE_RATIO = 0.08
GUIDE_EPS = 1e-4

# 투과율의 하한 (너무 작으면 색이 과하게 튄다)
T_MIN = 0.1

# 대기광을 정할 때 쓰는 상위 픽셀 비율
TOP_RATIO = 0.001


def dark_channel(image, size):
    # 픽셀마다 R, G, B 중 최솟값을 고르고, size x size 패치 안의 최솟값을 구한다.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(image.min(axis=2), kernel)


def guided_filter(guide, source, radius, eps):
    # guide(흑백 이미지)의 경계를 따라 source를 부드럽게 다듬는다.
    ksize = (radius, radius)
    mean_i = cv2.boxFilter(guide, cv2.CV_64F, ksize)
    mean_p = cv2.boxFilter(source, cv2.CV_64F, ksize)
    cov_ip = cv2.boxFilter(guide * source, cv2.CV_64F, ksize) - mean_i * mean_p
    var_i = cv2.boxFilter(guide * guide, cv2.CV_64F, ksize) - mean_i * mean_i

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, cv2.CV_64F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_64F, ksize)

    return mean_a * guide + mean_b


def Dehaze(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in OMEGA_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # 너무 작은 이미지는 그대로 돌려준다.
    if width < 3 or height < 3:
        return image.copy()

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # level별 범위 안에서 omega를 랜덤으로 결정
    omega = rng.uniform(*OMEGA_RANGES[level])

    short_side = min(width, height)
    patch = max(3, int(round(PATCH_RATIO * short_side)))
    radius = max(3, int(round(GUIDE_RATIO * short_side)))

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB"))
    hazy = rgb.astype(np.float64) / 255.0

    # 1. 다크 채널
    dark = dark_channel(hazy, patch)

    # 2. 대기광: 다크 채널 상위 0.1% 픽셀들의 평균 색
    count = max(1, int(dark.size * TOP_RATIO))
    brightest = np.argpartition(dark.ravel(), -count)[-count:]
    atmosphere = hazy.reshape(-1, 3)[brightest].mean(axis=0)
    atmosphere = np.maximum(atmosphere, 1e-3)

    # 3. 투과율
    transmission = 1.0 - omega * dark_channel(hazy / atmosphere, patch)

    # 4. 원본 밝기를 길잡이로 투과율을 다듬는다.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    transmission = guided_filter(gray, transmission, radius, GUIDE_EPS)
    transmission = np.maximum(transmission, T_MIN)

    # 5. 복원
    restored = (hazy - atmosphere) / transmission[..., None] + atmosphere

    Dehaze = Image.fromarray(
        np.clip(np.rint(restored * 255.0), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Dehaze.putalpha(alpha)

    return Dehaze
