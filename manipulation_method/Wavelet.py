import math
import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
Wavelet 변조 (랜덤 주파수 대역 웨이블릿 변환)

이미지를 웨이블릿 변환으로 여러 주파수 대역(scale)으로 쪼갠 뒤,
랜덤으로 고른 방식으로 특정 대역의 계수를 바꾸고 다시 이미지로 되돌린다.

웨이블릿 변환은 이미지를 "전체적인 형태(저주파)"와
"단계별 디테일(고주파 -> 중간 주파수)"로 나눈다.
  scale 1: 가장 잘게 쪼갠 디테일 (선의 가장자리, 글자 획, 잔 질감)
  scale 2: 조금 더 굵은 디테일
  scale 3, 4, ...: 점점 큰 범위의 명암 변화
각 scale에는 가로 / 세로 / 대각선 방향의 디테일 계수가 따로 있다.

웨이블릿 종류는 아래 두 가지 중에서 랜덤으로 고른다. (JPEG2000에서 쓰는 필터)
- CDF 9/7 (손실 압축용): 부드럽게 퍼지는 물결 모양 잔상(ringing)이 생긴다.
- LeGall 5/3 (무손실용): 조금 더 각지고 계단 같은 잔상이 생긴다.

계수를 바꾸는 방식은 아래 세 가지 중에서 랜덤으로 고른다.
- QUANTIZE: 모든 scale의 디테일 계수를 일정 간격(step)으로 반올림한다.
            작은 계수는 0이 되어 잔 질감이 사라지고, 선 주변에 물결 잔상이 생긴다.
            (JPEG2000으로 강하게 압축한 것과 같은 모습)
- LOWPASS:  가장 잘게 쪼갠 scale부터 몇 단계의 디테일 계수를 모두 0으로 지운다.
            고주파가 사라져 흐려지면서 선 주변에 웨이블릿 특유의 잔상이 남는다.
- BOOST:    고주파(scale 1~2) 또는 중간 주파수(scale 2~3) 대역 하나를 골라
            디테일 계수를 gain배 키운다. 그 굵기의 경계와 질감이 과장된다.

웨이블릿 종류와 방식은 이미지마다 랜덤으로 한 번 정해 lv1~lv4에 같이 쓰고,
level은 세 방식 모두에서 바꾸는 정도를 정한다.
지우는 scale 수는 짧은 변 720px 기준이며, 이미지가 2배 클 때마다 1단계씩 늘린다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

세 방식의 같은 level이 원본 대비 비슷한 정도(PSNR)로 손상되도록 맞춰 두었다.

level별 QUANTIZE step / LOWPASS 지우는 scale 수 / BOOST gain:
level 1: 40~60    / 1단계 / 1.5~2.0배
level 2: 80~120   / 2단계 / 2.0~2.8배
level 3: 150~220  / 3단계 / 2.8~3.8배
level 4: 280~400  / 4단계 / 3.8~5.0배

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 웨이블릿 설정
# quant_step:     QUANTIZE에서 디테일 계수를 반올림하는 간격
#                 (계수는 거친 scale일수록 커지므로 픽셀값 범위보다 큰 값도 쓴다)
# lowpass_scales: LOWPASS에서 지울 디테일 scale 수 (짧은 변 720px 기준)
# boost_gain:     BOOST에서 디테일 계수를 키우는 배율
CONFIGS = {
    1: {"quant_step": (40, 60), "lowpass_scales": 1, "boost_gain": (1.5, 2.0)},
    2: {"quant_step": (80, 120), "lowpass_scales": 2, "boost_gain": (2.0, 2.8)},
    3: {"quant_step": (150, 220), "lowpass_scales": 3, "boost_gain": (2.8, 3.8)},
    4: {"quant_step": (280, 400), "lowpass_scales": 4, "boost_gain": (3.8, 5.0)},
}

# 계수를 바꾸는 방식 후보
MODES = ["QUANTIZE", "LOWPASS", "BOOST"]

# 웨이블릿 종류 후보
WAVELETS = ["CDF97", "LEGALL53"]

# 웨이블릿 변환 단계 수 (이미지가 작으면 자동으로 줄인다)
MAX_DECOMPOSITION = 5

# scale 수를 정할 때 기준이 되는 짧은 변 길이
REFERENCE_SIDE = 720

# True면 웨이블릿 종류와 계수를 바꾸는 방식을(를) 이미지마다 한 번만 골라 lv1~lv4에 같이 쓴다. (강도만 level별로 다름)
# False면 level마다 따로 고른다.
SAME_STYLE_ALL_LEVELS = True

# ------------------------------------------------------------
# 리프팅(lifting) 방식 웨이블릿 계수
# (JPEG2000 표준 필터, 저주파/고주파의 세기가 비슷하도록 정규화)
# ------------------------------------------------------------
LIFTING_STEPS = {
    "CDF97": [
        (-1.586134342059924, "predict"),
        (-0.052980118572961, "update"),
        (0.882911075530934, "predict"),
        (0.443506852043971, "update"),
    ],
    "LEGALL53": [
        (-0.5, "predict"),
        (0.25, "update"),
    ],
}

LOW_SCALE = {"CDF97": 1.1496043988602418, "LEGALL53": math.sqrt(2)}
HIGH_SCALE = {"CDF97": 0.8698644516247808, "LEGALL53": 1 / math.sqrt(2)}


def lift_forward(x, axis, wavelet):
    # 한 축 방향으로 1단계 웨이블릿 변환 -> (저주파, 고주파)
    # 이미지 끝은 거울 반사로 이어 붙인 것처럼 계산한다.

    x = np.moveaxis(x, axis, 0)
    low = x[0::2].copy()
    high = x[1::2].copy()

    for coef, step in LIFTING_STEPS[wavelet]:
        if step == "predict":
            low_next = np.concatenate([low[1:], low[-1:]], axis=0)
            high += coef * (low + low_next)
        else:
            high_prev = np.concatenate([high[:1], high[:-1]], axis=0)
            low += coef * (high_prev + high)

    low *= LOW_SCALE[wavelet]
    high *= HIGH_SCALE[wavelet]

    return np.moveaxis(low, 0, axis), np.moveaxis(high, 0, axis)


def lift_inverse(low, high, axis, wavelet):
    # lift_forward의 역변환

    low = np.moveaxis(low, axis, 0) / LOW_SCALE[wavelet]
    high = np.moveaxis(high, axis, 0) / HIGH_SCALE[wavelet]

    for coef, step in reversed(LIFTING_STEPS[wavelet]):
        if step == "predict":
            low_next = np.concatenate([low[1:], low[-1:]], axis=0)
            high -= coef * (low + low_next)
        else:
            high_prev = np.concatenate([high[:1], high[:-1]], axis=0)
            low -= coef * (high_prev + high)

    merged = np.empty((low.shape[0] * 2,) + low.shape[1:], dtype=low.dtype)
    merged[0::2] = low
    merged[1::2] = high

    return np.moveaxis(merged, 0, axis)


def dwt2(x, wavelet, levels):
    # 2차원 웨이블릿 변환 -> (가장 거친 저주파, [scale 1의 디테일, scale 2의 디테일, ...])

    details = []
    approx = x

    for _ in range(levels):
        low, high = lift_forward(approx, 1, wavelet)
        low_low, low_high = lift_forward(low, 0, wavelet)
        high_low, high_high = lift_forward(high, 0, wavelet)
        details.append([low_high, high_low, high_high])
        approx = low_low

    return approx, details


def idwt2(approx, details, wavelet):
    # dwt2의 역변환

    for low_high, high_low, high_high in reversed(details):
        low = lift_inverse(approx, low_high, 0, wavelet)
        high = lift_inverse(high_low, high_high, 0, wavelet)
        approx = lift_inverse(low, high, 1, wavelet)

    return approx


def Wavelet(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in CONFIGS:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # 웨이블릿 변환을 1단계도 할 수 없을 만큼 작은 이미지는 그대로 돌려준다.
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

    # 웨이블릿 종류, 계수를 바꾸는 방식, BOOST할 대역은 이미지마다 랜덤으로 결정
    wavelet = style_rng.choice(WAVELETS)
    mode = style_rng.choice(MODES)
    boost_start = style_rng.choice([0, 1])

    # 이미지 크기에 맞춰 변환 단계 수를 정한다. (가장 거친 단계도 4px 이상)
    levels = max(1, min(MAX_DECOMPOSITION, int(math.log2(short_side)) - 2))

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB"))

    # 변환 단계 수만큼 2로 나누어떨어지도록 가장자리를 거울 반사로 늘린다.
    block = 2 ** levels
    pad_bottom = (-height) % block
    pad_right = (-width) % block
    padded = cv2.copyMakeBorder(
        rgb, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101
    ).astype(np.float64)

    approx, details = dwt2(padded, wavelet, levels)

    if mode == "QUANTIZE":
        # 모든 scale의 디테일 계수를 step 간격으로 반올림한다. (0 근처는 0으로)
        step = rng.uniform(*config["quant_step"])
        for bands in details:
            for idx, band in enumerate(bands):
                q = np.sign(band) * np.floor(np.abs(band) / step)
                bands[idx] = np.where(q != 0, (q + 0.5 * np.sign(q)) * step, 0.0)

    elif mode == "LOWPASS":
        # 해상도에 맞춰 지울 scale 수를 정한다. (2배 클 때마다 1단계 추가)
        extra = max(0, round(math.log2(short_side / REFERENCE_SIDE)))
        scales = min(levels, config["lowpass_scales"] + extra)
        for scale in range(scales):
            details[scale] = [np.zeros_like(band) for band in details[scale]]

    else:  # BOOST
        # 고주파(scale 1~2) 또는 중간 주파수(scale 2~3) 대역을 키운다.
        gain = rng.uniform(*config["boost_gain"])
        for scale in range(boost_start, min(levels, boost_start + 2)):
            details[scale] = [band * gain for band in details[scale]]

    restored = idwt2(approx, details, wavelet)[:height, :width]

    Wavelet = Image.fromarray(
        np.clip(np.rint(restored), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Wavelet.putalpha(alpha)

    return Wavelet
