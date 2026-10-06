import cv2
import numpy as np
from PIL import Image

"""
CorrectExpo 변조 (노출 보정)

너무 어둡게 찍힌 곳은 밝히고 너무 밝게 찍힌 곳은 어둡게 눌러,
이미지 전체의 노출을 고르게 맞춘다.
결과는 HDR 사진처럼 어두운 곳과 밝은 곳의 디테일이 함께 드러나고,
밝은 하늘은 어두워지며 질감이 강조된다.

1. 조명 추정: 픽셀마다 R, G, B 중 가장 큰 값을 그 위치의 밝기로 보고,
   이를 넓게 흐려 "그 주변이 얼마나 밝게 찍혔는지"를 나타내는 조명 지도를 만든다.
2. 노출 부족 보정: 이미지를 조명 지도로 나눈다.
   조명이 어두운 곳일수록 더 많이 밝아진다.
3. 노출 과다 보정: 이미지를 반전시켜 2번과 같은 보정을 한 뒤 다시 반전시킨다.
   조명이 밝은 곳일수록 더 많이 어두워진다.
4. 합성: 원본, 2번 결과, 3번 결과를 노출 합성(Mertens exposure fusion)으로 합친다.
   픽셀마다 세 장 중 대비와 채도가 좋고 밝기가 적당한 쪽을 더 많이 반영한다.

level이 높을수록 2, 3번의 보정을 강하게 한다.
랜덤 요소가 없어, 같은 입력과 level에 대해 항상 같은 결과가 나온다.

출력은 RGB이고(투명도가 있는 이미지는 RGBA), 출력 크기는 원본과 같다.

level별 보정 강도 (0이면 보정 없음, 1이면 조명 차이를 완전히 없앰):
level 1: 0.3
level 2: 0.5
level 3: 0.7
level 4: 0.9
"""

# level별 보정 강도
STRENGTHS = {
    1: 0.3,
    2: 0.5,
    3: 0.7,
    4: 0.9,
}

# 조명 지도를 흐리는 정도 (이미지 짧은 변 대비)
BLUR_RATIO = 0.03

# 조명 값의 하한 (아주 어두운 곳을 나눌 때 값이 폭주하지 않게 한다)
MIN_ILLUMINATION = 0.05


def correct_under_exposure(rgb, strength, sigma):
    # 조명이 어두운 곳을 밝힌다. (rgb는 0~1 범위의 float 배열)

    # 픽셀마다 R, G, B 중 가장 큰 값을 밝기로 보고, 넓게 흐려 조명 지도를 만든다.
    illumination = cv2.GaussianBlur(rgb.max(axis=2), (0, 0), sigma)
    illumination = np.clip(illumination, MIN_ILLUMINATION, 1.0)

    # 조명 지도로 나눈다. strength가 클수록 조명 차이를 더 많이 없앤다.
    corrected = rgb / (illumination ** strength)[..., None]

    return np.clip(corrected, 0.0, 1.0)


def CorrectExpo(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    **kwargs
) -> Image.Image:

    if level not in STRENGTHS:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # 너무 작은 이미지는 주변 밝기를 구할 수 없으므로 그대로 돌려준다.
    if width < 2 or height < 2:
        return image.copy()

    strength = STRENGTHS[level]
    sigma = max(1.0, BLUR_RATIO * min(width, height))

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    # 0~1 범위로 정규화한 RGB 배열
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    # 노출 부족 보정: 어두운 곳을 밝힌다.
    under_fixed = correct_under_exposure(rgb, strength, sigma)

    # 노출 과다 보정: 반전시켜 같은 보정을 한 뒤 다시 반전시킨다.
    over_fixed = 1.0 - correct_under_exposure(1.0 - rgb, strength, sigma)

    # 원본과 두 보정 결과를 노출 합성으로 합친다.
    exposures = [
        np.clip(np.rint(layer * 255.0), 0, 255).astype(np.uint8)
        for layer in (rgb, under_fixed, over_fixed)
    ]

    fused = cv2.createMergeMertens().process(exposures)

    CorrectExpo = Image.fromarray(
        np.clip(np.rint(fused * 255.0), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        CorrectExpo.putalpha(alpha)

    return CorrectExpo