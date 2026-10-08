# 색상 수를 줄이고 검은 윤곽선을 강조하여 단순화된 cartoon style로 변환하는 코드

from PIL import Image
import numpy as np
import cv2


def Cartoonize(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    arr = np.array(
        image.convert("RGB")
    )

    # 색상 smoothing
    smooth = cv2.bilateralFilter(
        arr,
        d=9,
        sigmaColor=100,
        sigmaSpace=100
    )

    # color quantization/ div가 커질수록 색상 단계가 줄어들어서 더 단순한 cartoon 느낌 남
    div = {
        1: 30,
        2: 40,
        3: 50,
        4: 60,
    }[level]

    quantized = (
        smooth // div
    ) * div

    gray = cv2.cvtColor(
        arr,
        cv2.COLOR_RGB2GRAY
    )

    edge = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        8
    )

    edge = cv2.cvtColor(
        edge,
        cv2.COLOR_GRAY2RGB
    )

    result = cv2.bitwise_and(
        quantized,
        edge
    )

    return Image.fromarray(result)