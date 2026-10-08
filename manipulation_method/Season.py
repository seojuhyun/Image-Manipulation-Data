# 색상 채널을 조절하여 spring, summer, autumn, winter 계절 분위기로 변환하는 코드

from PIL import Image
import numpy as np


def Season(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    arr = np.array(
        image.convert("RGB"),
        dtype=np.float32
    )

    if level == 1:
        # Spring
        arr[..., 0] *= 0.95
        arr[..., 1] *= 1.12
        arr[..., 2] *= 1.12

    elif level == 2:
        # Summer
        arr[..., 0] *= 1.12
        arr[..., 1] *= 1.08
        arr[..., 2] *= 0.90

    elif level == 3:
        # Autumn
        arr[..., 0] *= 1.20
        arr[..., 1] *= 0.90
        arr[..., 2] *= 1.05

    else:
        # Winter
        arr[..., 0] *= 0.90
        arr[..., 1] *= 0.98
        arr[..., 2] *= 1.20

    return Image.fromarray(
        np.clip(
            arr,
            0,
            255
        ).astype(np.uint8)
    )