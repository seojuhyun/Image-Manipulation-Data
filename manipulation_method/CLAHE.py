import cv2
import numpy as np
from PIL import Image

"""
CLAHE 변조 (대비 제한 적응형 히스토그램 평활화)

이미지의 국소 대비를 높여, 어두운 곳과 밝은 곳의 디테일을 또렷하게 만든다.
밋밋하던 영역에도 질감이 드러나 HDR 사진 같은 느낌이 난다.

1. 이미지를 LAB 색 공간으로 바꿔 밝기(L) 채널만 꺼낸다.
   (색이 틀어지지 않도록 색 채널은 건드리지 않는다.)
2. 이미지를 격자로 나누고, 칸마다 밝기 분포를 고르게 펴서
   그 칸 안에서 어두운 값부터 밝은 값까지 넓게 쓰이도록 한다.
   이때 한 밝기에 픽셀이 지나치게 몰리지 않도록 상한(clip limit)을 둔다.
   상한이 클수록 대비가 강해지고, 노이즈도 함께 두드러진다.
3. 칸 경계가 보이지 않도록 이웃한 칸의 결과를 부드럽게 섞는다.

랜덤 요소가 없어, 같은 입력과 level에 대해 항상 같은 결과가 나온다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 clip limit / 격자 수 (이미지 짧은 변 기준):
level 1: 2.0  / 8칸
level 2: 4.0  / 8칸
level 3: 6.0  / 16칸
level 4: 10.0 / 16칸
"""

# level별 CLAHE 설정
# clip_limit: 대비를 얼마나 강하게 펼지 (클수록 강함)
# grid_cells: 이미지 짧은 변을 몇 칸으로 나눌지 (클수록 좁은 범위 기준으로 평활화)
CONFIGS = {
    1: {"clip_limit": 2.0, "grid_cells": 8},
    2: {"clip_limit": 4.0, "grid_cells": 8},
    3: {"clip_limit": 6.0, "grid_cells": 16},
    4: {"clip_limit": 10.0, "grid_cells": 16},
}


def CLAHE(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    **kwargs
) -> Image.Image:

    if level not in CONFIGS:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    config = CONFIGS[level]
    width, height = image.size

    # 격자 칸이 정사각형에 가깝도록 가로/세로 칸 수를 정한다.
    cell_size = max(1.0, min(width, height) / config["grid_cells"])
    cols = max(1, round(width / cell_size))
    rows = max(1, round(height / cell_size))

    clahe = cv2.createCLAHE(
        clipLimit=config["clip_limit"],
        tileGridSize=(cols, rows)
    )

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    if image.mode == "L":
        # 흑백 이미지는 밝기 값에 바로 적용한다.
        equalized = clahe.apply(np.asarray(image))

    else:
        rgb = np.asarray(image.convert("RGB"))

        # 밝기(L) 채널에만 적용하고 색(A, B) 채널은 그대로 둔다.
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    CLAHE = Image.fromarray(equalized)

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        CLAHE.putalpha(alpha)

    return CLAHE