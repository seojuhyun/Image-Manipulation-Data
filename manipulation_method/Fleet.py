# 색조와 대비를 변화시켜 시간의 흐름이나 오래된 분위기를 표현하는 Fleet 코드

from PIL import Image, ImageEnhance
import numpy as np


def Fleet(
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
        # purple / blue cast
        arr[..., 0] *= 1.10
        arr[..., 1] *= 0.85
        arr[..., 2] *= 1.30

    elif level == 2:
        # aged yellow-green
        arr[..., 0] *= 1.10
        arr[..., 1] *= 1.15
        arr[..., 2] *= 0.70

    elif level == 3:
        # strong cool blue
        arr[..., 0] *= 0.90
        arr[..., 1] *= 0.90
        arr[..., 2] *= 1.45

    else:
        # faded blue-purple
        arr[..., 0] *= 1.00
        arr[..., 1] *= 0.85
        arr[..., 2] *= 1.35

    arr = np.clip(
        arr,
        0,
        255
    ).astype(np.uint8)

    result = Image.fromarray(arr)

    result = ImageEnhance.Contrast(
        result
    ).enhance(1.15)

    return result