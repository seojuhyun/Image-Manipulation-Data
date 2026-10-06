import numpy as np
from PIL import Image

"""
Sepia 변조 (세피아 톤 변환)

이미지를 오래된 사진처럼 갈색빛이 도는 세피아 톤으로 바꾼다.

각 픽셀의 RGB 값에 표준 세피아 변환 행렬을 곱해 세피아 색을 구한 뒤,
level별 비율로 원본과 섞는다.
비율이 100%면 완전한 세피아 톤이 되고, 낮을수록 원본 색이 남는다.
랜덤 요소가 없어, 같은 입력과 level에 대해 항상 같은 결과가 나온다.

출력은 RGB이고(투명도가 있는 이미지는 RGBA), 출력 크기는 원본과 같다.

level별 세피아 혼합 비율:
level 1: 25%
level 2: 50%
level 3: 75%
level 4: 100%
"""

# level별 세피아 혼합 비율
BLEND_FACTORS = {
    1: 0.25,
    2: 0.50,
    3: 0.75,
    4: 1.00,
}

# 표준 세피아 변환 행렬
# R' = 0.393R + 0.769G + 0.189B
# G' = 0.349R + 0.686G + 0.168B
# B' = 0.272R + 0.534G + 0.131B
SEPIA_MATRIX = np.array(
    [
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ],
    dtype=np.float32
)


def Sepia(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    **kwargs
) -> Image.Image:

    if level not in BLEND_FACTORS:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    factor = BLEND_FACTORS[level]

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)

    # 행렬 곱으로 세피아 색을 계산 (255를 넘는 값은 잘라낸다)
    sepia = np.clip(rgb @ SEPIA_MATRIX.T, 0, 255)

    # level별 비율로 원본과 세피아 색을 섞는다.
    blended = (1 - factor) * rgb + factor * sepia

    Sepia = Image.fromarray(
        np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Sepia.putalpha(alpha)

    return Sepia