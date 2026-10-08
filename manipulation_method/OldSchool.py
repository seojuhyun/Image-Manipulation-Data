# level별로 서로 다른 vintage/retro 효과를 적용하여 old-school style을 생성하는 코드
'''
from PIL import Image
import numpy as np
import cv2


def OldSchool(
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

    h, w = arr.shape[:2]

    # ========================================================
    # level 1: Sepia / old film
    # ========================================================
    if level == 1:

        gray = (
            0.299 * arr[..., 0]
            + 0.587 * arr[..., 1]
            + 0.114 * arr[..., 2]
        )

        result = np.stack([
            gray * 1.20,
            gray * 1.00,
            gray * 0.75
        ], axis=-1)

    # ========================================================
    # level 2: Green monochrome terminal
    # ========================================================
    elif level == 2:

        gray = cv2.cvtColor(
            arr.astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        ).astype(np.float32)

        result = np.zeros_like(arr)

        result[..., 0] = gray * 0.20
        result[..., 1] = gray * 1.20
        result[..., 2] = gray * 0.35

        # horizontal scanline
        result[::4] *= 0.55

    # ========================================================
    # level 3: CRT television
    # ========================================================
    elif level == 3:

        result = arr.copy()

        yy, xx = np.indices((h, w))

        cx = w / 2
        cy = h / 2

        dx = (xx - cx) / cx
        dy = (yy - cy) / cy

        dist = np.sqrt(
            dx ** 2 + dy ** 2
        )

        vignette = np.clip(
            1.20 - dist * 0.85,
            0.20,
            1.0
        )

        result *= vignette[..., None]

        # stronger scanlines
        result[::3] *= 0.55

        # CRT color shift
        result[..., 0] *= 1.05
        result[..., 1] *= 0.95
        result[..., 2] *= 0.85

    # ========================================================
    # level 4: faded VHS / blue-purple retro
    # ========================================================
    else:

        result = arr.copy()

        # faded color
        result[..., 0] *= 1.05
        result[..., 1] *= 0.75
        result[..., 2] *= 1.35

        # contrast 감소
        mean = result.mean(
            axis=(0, 1),
            keepdims=True
        )

        result = (
            mean
            + 0.70 * (result - mean)
        )

        # horizontal VHS noise
        for y in range(0, h, 7):
            result[y:y + 1] *= 0.65

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(result)
'''

# old film, green terminal, CRT, strong CRT의 4가지 old-school variant를 적용하는 코드

from PIL import Image
import numpy as np
import cv2


def OldSchool(
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

    h, w = arr.shape[:2]

    # ========================================================
    # level 1: Old film / Sepia
    # ========================================================
    if level == 1:

        gray = (
            0.299 * arr[..., 0]
            + 0.587 * arr[..., 1]
            + 0.114 * arr[..., 2]
        )

        result = np.stack([
            gray * 1.20,
            gray * 1.00,
            gray * 0.75
        ], axis=-1)

    # ========================================================
    # level 2: Green monochrome terminal
    # ========================================================
    elif level == 2:

        gray = cv2.cvtColor(
            arr.astype(np.uint8),
            cv2.COLOR_RGB2GRAY
        ).astype(np.float32)

        result = np.zeros_like(arr)

        result[..., 0] = gray * 0.20
        result[..., 1] = gray * 1.20
        result[..., 2] = gray * 0.35

        # terminal scanline
        result[::4] *= 0.55

    # ========================================================
    # level 3: CRT television
    # ========================================================
    elif level == 3:

        result = arr.copy()

        yy, xx = np.indices((h, w))

        cx = w / 2
        cy = h / 2

        dx = (xx - cx) / cx
        dy = (yy - cy) / cy

        dist = np.sqrt(
            dx ** 2 + dy ** 2
        )

        # mild vignette
        vignette = np.clip(
            1.20 - dist * 0.75,
            0.30,
            1.0
        )

        result *= vignette[..., None]

        # CRT scanlines
        result[::4] *= 0.70

        # 약한 CRT color shift
        result[..., 0] *= 1.05
        result[..., 1] *= 0.95
        result[..., 2] *= 0.90

    # ========================================================
    # level 4: Strong CRT television
    # ========================================================
    else:

        result = arr.copy()

        yy, xx = np.indices((h, w))

        cx = w / 2
        cy = h / 2

        dx = (xx - cx) / cx
        dy = (yy - cy) / cy

        dist = np.sqrt(
            dx ** 2 + dy ** 2
        )

        # stronger vignette
        vignette = np.clip(
            1.15 - dist * 1.00,
            0.10,
            1.0
        )

        result *= vignette[..., None]

        # stronger / denser scanlines
        result[::3] *= 0.45

        # CRT color separation 느낌
        shifted = result.copy()

        shift = max(1, w // 300)

        shifted[..., 0] = np.roll(
            result[..., 0],
            shift,
            axis=1
        )

        shifted[..., 2] = np.roll(
            result[..., 2],
            -shift,
            axis=1
        )

        result = shifted

        # contrast 강화
        mean = result.mean(
            axis=(0, 1),
            keepdims=True
        )

        result = (
            mean
            + 1.15 * (result - mean)
        )

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(result)