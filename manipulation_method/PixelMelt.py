# 이미지의 여러 구간을 아래 방향으로 길게 늘여 pixel melting 효과를 생성하는 코드

from PIL import Image
import numpy as np


def PixelMelt(
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

    h, w = arr.shape[:2]

    result = arr.copy()

    num_strips = {
        1: 10,
        2: 16,
        3: 23,
        4: 30,
    }[level]

    max_length = {
        1: int(h * 0.12),
        2: int(h * 0.20),
        3: int(h * 0.30),
        4: int(h * 0.40),
    }[level]

    for _ in range(num_strips):

        strip_w = int(
            rng.integers(
                max(2, w // 100),
                max(3, w // 20)
            )
        )

        x = int(
            rng.integers(
                0,
                max(1, w - strip_w)
            )
        )

        y = int(
            rng.integers(
                0,
                max(1, h - 2)
            )
        )

        length = int(
            rng.integers(
                5,
                max(6, max_length)
            )
        )

        end_y = min(
            h,
            y + length
        )

        # source strip를 아래로 늘임
        source = result[
            y:y + 1,
            x:x + strip_w
        ]

        result[
            y:end_y,
            x:x + strip_w
        ] = np.repeat(
            source,
            end_y - y,
            axis=0
        )

    return Image.fromarray(result)