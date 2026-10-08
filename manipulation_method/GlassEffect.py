# 국소적인 pixel displacement와 blur를 적용하여 glass distortion 효과를 생성하는 코드

from PIL import Image
import numpy as np
import cv2


def GlassEffect(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    rng = np.random.default_rng(seed)

    arr = np.array(
        image.convert("RGB")
    )

    height, width = arr.shape[:2]

    radius = {
        1: 2,
        2: 3.8,
        3: 5.6,
        4: 7.4,
    }[level]

    yy, xx = np.indices(
        (height, width)
    )

    offset_x = rng.normal(
        0,
        radius,
        (height, width)
    ).astype(np.int32)

    offset_y = rng.normal(
        0,
        radius,
        (height, width)
    ).astype(np.int32)

    map_x = np.clip(
        xx + offset_x,
        0,
        width - 1
    ).astype(np.float32)

    map_y = np.clip(
        yy + offset_y,
        0,
        height - 1
    ).astype(np.float32)

    distorted = cv2.remap(
        arr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR
    )

    blur = {
        1: 3,
        2: 3,
        3: 5,
        4: 5,
    }[level]

    distorted = cv2.GaussianBlur(
        distorted,
        (blur, blur),
        0
    )

    return Image.fromarray(distorted)