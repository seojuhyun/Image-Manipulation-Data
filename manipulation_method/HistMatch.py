import random
import zlib

import numpy as np
from PIL import Image

"""
HistMatch 변조 (랜덤 히스토그램 매칭)

이미지의 밝기 분포(히스토그램)가 랜덤하게 만든 목표 분포와 같아지도록
픽셀 값을 바꾼다. R, G, B 채널마다 서로 다른 목표 분포를 쓰기 때문에,
채널별 밝기가 따로 움직여 이미지 전체에 색조가 입혀진다.
(예: 붉은 채널은 밝게, 푸른 채널은 어둡게 몰리면 전체가 분홍빛이 된다.)

채널마다 아래 과정을 따로 수행한다.
1. 목표 분포 만들기: 랜덤한 위치와 폭을 가진 종 모양 봉우리 1~2개를 겹쳐
   "어떤 밝기의 픽셀이 얼마나 많아야 하는지"를 정한다.
2. 매칭: 원본에서 어두운 쪽부터 세어 하위 몇 %에 해당하는 픽셀은,
   목표 분포에서도 하위 같은 %에 해당하는 밝기로 바꾼다.
   픽셀들의 밝기 순서는 그대로 유지되므로 형태는 변하지 않는다.
3. 섞기: 매칭 결과를 level별 비율로 원본과 섞는다.

비율이 100%면 분포가 목표와 완전히 같아지고, 낮을수록 원본에 가깝다.
목표 분포의 봉우리가 좁으면 여러 밝기가 비슷한 값으로 몰려,
하늘 같은 부드러운 영역에 띠 모양 경계가 생길 수 있다.

출력은 RGB이고(투명도가 있는 이미지는 RGBA), 출력 크기는 원본과 같다.

level별 매칭 결과를 섞는 비율:
level 1: 25%
level 2: 50%
level 3: 75%
level 4: 100%

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 매칭 결과를 섞는 비율
BLEND_FACTORS = {
    1: 0.25,
    2: 0.50,
    3: 0.75,
    4: 1.00,
}

# 목표 분포의 봉우리 위치 범위 (0 = 검은색, 1 = 흰색)
PEAK_CENTER_RANGE = (0.15, 0.85)

# 목표 분포의 봉우리 폭(표준편차) 범위
PEAK_WIDTH_RANGE = (0.08, 0.30)


def random_histogram(rng):
    # 랜덤한 목표 분포를 만든다. (길이 256, 합이 1인 배열)

    levels = np.arange(256, dtype=np.float64) / 255.0
    histogram = np.zeros(256, dtype=np.float64)

    # 종 모양 봉우리 1~2개를 랜덤한 위치, 폭, 높이로 겹친다.
    for _ in range(rng.randint(1, 2)):
        center = rng.uniform(*PEAK_CENTER_RANGE)
        width = rng.uniform(*PEAK_WIDTH_RANGE)
        weight = rng.uniform(0.5, 1.0)

        histogram += weight * np.exp(-0.5 * ((levels - center) / width) ** 2)

    return histogram / histogram.sum()


def match_channel(channel, target_histogram):
    # 한 채널의 밝기 분포가 target_histogram과 같아지도록 값을 바꾼다.

    # 원본의 누적 분포: 각 밝기 값 이하인 픽셀의 비율
    source_histogram = np.bincount(channel.ravel(), minlength=256)
    source_cdf = np.cumsum(source_histogram) / channel.size

    # 목표의 누적 분포
    target_cdf = np.cumsum(target_histogram)

    # 원본의 각 밝기 값이 하위 몇 %인지 보고,
    # 목표 분포에서 같은 %에 해당하는 밝기 값을 찾는다.
    table = np.interp(source_cdf, target_cdf, np.arange(256))

    return table[channel]


def HistMatch(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in BLEND_FACTORS:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    factor = BLEND_FACTORS[level]

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB"))
    matched = np.zeros(rgb.shape, dtype=np.float64)

    # 채널마다 서로 다른 랜덤 목표 분포에 맞춘다.
    for idx in range(3):
        matched[..., idx] = match_channel(
            rgb[..., idx],
            random_histogram(rng)
        )

    # level별 비율로 원본과 매칭 결과를 섞는다.
    blended = (1 - factor) * rgb + factor * matched

    HistMatch = Image.fromarray(
        np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        HistMatch.putalpha(alpha)

    return HistMatch