import random
import zlib

import numpy as np
from PIL import Image

"""
FancyPCA 변조 (PCA 기반 랜덤 색상 변화)

이미지의 모든 픽셀에 같은 RGB 값을 더해 전체 색감과 밝기를 바꾼다.
더하는 값의 방향은 그 이미지의 색 분포에서 PCA로 구한다.
(Krizhevsky et al., 2012의 색상 증강 방식)

1. 이미지의 모든 픽셀을 RGB 3차원 공간의 점으로 보고,
   색이 가장 많이 퍼져 있는 방향 세 개(주성분)와
   각 방향으로 퍼진 정도(표준편차)를 구한다.
2. 주성분마다 랜덤한 배율을 정하고,
   (배율 x 표준편차 x 주성분 방향)을 모두 더해 RGB 이동량을 만든다.
3. 이 이동량을 모든 픽셀에 똑같이 더한다.

이미지가 원래 갖고 있는 색 변화의 방향을 따라 움직이므로,
아무 색이나 덧씌우는 것보다 자연스러운 조명/색감 변화처럼 보인다.
배율이 크면 이미지 전체가 한쪽 색으로 쏠리거나 하얗게/어둡게 날아간다.

배율의 크기는 level별 범위 안에서, 방향(부호)은 랜덤으로 정한다.
출력은 RGB이고(투명도가 있는 이미지는 RGBA), 출력 크기는 원본과 같다.

level별 배율 범위 (주성분 방향 표준편차의 몇 배만큼 옮길지):
level 1: 0.2~0.4배
level 2: 0.4~0.8배
level 3: 0.8~1.2배
level 4: 1.2~1.8배

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 배율 범위 (주성분 방향 표준편차의 몇 배만큼 옮길지)
ALPHA_RANGES = {
    1: (0.2, 0.4),
    2: (0.4, 0.8),
    3: (0.8, 1.2),
    4: (1.2, 1.8),
}


def FancyPCA(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in ALPHA_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # 픽셀이 2개 미만이면 색 분포를 구할 수 없으므로 그대로 돌려준다.
    if width * height < 2:
        return image.copy()

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha_channel = image.getchannel("A") if has_alpha else None

    # 0~1 범위로 정규화한 RGB 배열
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    # 픽셀을 (픽셀 수, 3) 형태로 늘어놓는다.
    pixels = rgb.reshape(-1, 3)

    # PCA: RGB 공분산 행렬의 고유값(분산)과 고유벡터(주성분 방향)
    cov = np.cov(pixels, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    # 각 주성분 방향의 표준편차 (계산 오차로 생긴 음수는 0으로 처리)
    stds = np.sqrt(np.clip(eig_vals, 0, None))

    # 고유벡터의 부호는 계산 환경에 따라 뒤집힐 수 있으므로,
    # 절댓값이 가장 큰 성분이 양수가 되도록 통일한다.
    for idx in range(3):
        vec = eig_vecs[:, idx]
        if vec[np.argmax(np.abs(vec))] < 0:
            eig_vecs[:, idx] = -vec

    # 주성분마다 배율을 랜덤으로 결정 (크기는 level별 범위 안에서, 부호는 랜덤)
    def random_alpha():
        value = rng.uniform(*ALPHA_RANGES[level])
        return value if rng.random() < 0.5 else -value

    alphas = np.array([random_alpha() for _ in range(3)])

    # 모든 픽셀에 더할 RGB 이동량
    # = 주성분 방향 x (배율 x 표준편차) 를 세 주성분에 대해 합한 값
    offset = eig_vecs @ (alphas * stds)

    shifted = (rgb + offset.astype(np.float32)) * 255.0

    FancyPCA = Image.fromarray(
        np.clip(np.rint(shifted), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        FancyPCA.putalpha(alpha_channel)

    return FancyPCA