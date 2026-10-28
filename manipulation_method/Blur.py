import random
import zlib

import cv2
import numpy as np
from PIL import Image

"""
Blur 변조 (랜덤 가우시안 블러)

이미지에 가우시안 필터를 적용해 흐리게 만든다.
각 픽셀을 주변 픽셀들과 종 모양(가우시안) 가중치로 평균 내므로,
선과 글자의 경계가 번지고 세밀한 질감이 사라진다.

블러의 세기(가우시안의 표준편차 sigma)는 level별 범위 안에서 랜덤으로 정한다.
sigma는 이미지 크기에 비례하도록 짧은 변 대비 비율로 정해,
해상도가 달라도 비슷한 정도로 흐려지게 한다.

lv1~lv2는 살짝 초점이 나간 정도의 약한 블러이고,
lv3~lv4는 SSCD의 Strong Blur처럼 반경이 큰 가우시안 블러로,
말풍선 글자를 읽기 어려울 정도로 강하게 흐려진다.
("Blur (Gaussian, Strong)" 항목도 이 기법의 lv3~lv4로 함께 다룬다.)

이미지 밖은 거울 반사로 채워 가장자리가 어두워지지 않는다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 sigma 범위 (이미지 짧은 변 대비 / 짧은 변 720px 기준):
level 1: 0.2~0.3% / 약 1.4~2.2px
level 2: 0.4~0.6% / 약 2.9~4.3px
level 3: 0.8~1.2% / 약 5.8~8.6px   (strong)
level 4: 1.6~2.4% / 약 11.5~17.3px (strong)

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 가우시안 sigma 범위 (이미지 짧은 변 대비)
SIGMA_RANGES = {
    1: (0.002, 0.003),
    2: (0.004, 0.006),
    3: (0.008, 0.012),
    4: (0.016, 0.024),
}


def Blur(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in SIGMA_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # 너무 작은 이미지는 흐리게 할 수 없으므로 그대로 돌려준다.
    if width < 2 or height < 2:
        return image.copy()

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # level별 범위 안에서 sigma를 랜덤으로 결정 (짧은 변 대비 비율 -> 픽셀)
    sigma = rng.uniform(*SIGMA_RANGES[level]) * min(width, height)
    sigma = max(0.3, sigma)

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB"))

    # 커널 크기는 sigma에 맞춰 OpenCV가 자동으로 정한다. (약 6 x sigma)
    blurred = cv2.GaussianBlur(
        rgb,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101
    )

    Blur = Image.fromarray(blurred)

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Blur.putalpha(alpha)

    return Blur
