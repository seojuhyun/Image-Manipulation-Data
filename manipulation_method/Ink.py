# grayscale과 edge 정보를 결합하여 ink drawing 또는 ink painting 형태로 변환하는 코드

from PIL import Image
import numpy as np
import cv2


def Ink(
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

    gray = cv2.cvtColor(
        arr,
        cv2.COLOR_RGB2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        {
            1: 1.5,
            2: 4.0,
            3: 6.5,
            4: 9,
        }[level]
    )

    # grayscale tone 일부 유지
    alpha = {
        1: 0.25,
        2: 0.35,
        3: 0.45,
        4: 0.55,
    }[level]

    result = (
        alpha * gray
        + (1 - alpha) * edges
    )

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        cv2.cvtColor(
            result,
            cv2.COLOR_GRAY2RGB
        )
    )