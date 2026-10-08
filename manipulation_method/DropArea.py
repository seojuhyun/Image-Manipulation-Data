# lv1~2는 pixel dropout, lv3~4는 rectangular DropBlock을 적용하는 DropArea 코드
'''
lv1 → 색상 왜곡 + 작은 DropBlock
lv2 → 색상 왜곡 + 큰 DropBlock 여러 개
lv3 → 중간 수준 Dropout + 노이즈
lv4 → 강한 Dropout + 강한 노이즈/색 왜곡
'''
from PIL import Image, ImageDraw
import numpy as np
import random
import math
from manipulation_method.helper_overlay import (_get_default_font, _random_color, _random_pastel_color, _alpha_composite, _random_string, _draw_centered_text, _ensure_rgba)


def _add_salt_pepper_dropout(arr, drop_ratio, rng):
    """
    개별 pixel을 독립적으로 손상시키는 dropout 형태.
    일부는 검정, 일부는 흰색 또는 랜덤 색으로 바꾼다.
    """
    h, w = arr.shape[:2]
    total = h * w
    num_drop = int(total * drop_ratio)

    ys = rng.integers(0, h, size=num_drop)
    xs = rng.integers(0, w, size=num_drop)

    mode = rng.choice(["black", "white", "random"])

    if mode == "black":
        arr[ys, xs] = 0
    elif mode == "white":
        arr[ys, xs] = 255
    else:
        arr[ys, xs] = rng.integers(0, 256, size=(num_drop, 3))

    return arr


def _add_dropblock(arr, block_ratio, num_blocks, rng):
    """
    연속된 사각형 영역을 제거하는 dropblock 형태.
    """
    h, w = arr.shape[:2]
    target_area = h * w * block_ratio
    each_area = target_area / num_blocks

    for _ in range(num_blocks):
        aspect = rng.uniform(0.6, 1.8)
        bw = int(np.sqrt(each_area * aspect))
        bh = int(each_area / max(1, bw))

        bw = max(8, min(w, bw))
        bh = max(8, min(h, bh))

        x = rng.integers(0, max(1, w - bw + 1))
        y = rng.integers(0, max(1, h - bh + 1))

        arr[y:y + bh, x:x + bw] = 0

    return arr


def _false_color_shift(arr, mode):
    """
    예시처럼 초록/분홍/파랑 쪽으로 과장된 색 왜곡을 준다.
    """
    arr = arr.astype(np.float32)

    if mode == 1:
        # yellow-green sky 느낌
        arr[..., 0] *= 1.15
        arr[..., 1] *= 1.35
        arr[..., 2] *= 0.65

    elif mode == 2:
        # blue-magenta noisy 느낌
        arr[..., 0] *= 1.10
        arr[..., 1] *= 0.75
        arr[..., 2] *= 1.35

    elif mode == 3:
        # stronger red/green split
        arr[..., 0] *= 1.35
        arr[..., 1] *= 0.95
        arr[..., 2] *= 0.75

    elif mode == 4:
        # hot pink / green-like distortion
        arr[..., 0] *= 1.40
        arr[..., 1] *= 0.95
        arr[..., 2] *= 1.20

    return np.clip(arr, 0, 255).astype(np.uint8)


def _add_gaussian_noise(arr, sigma, rng):
    """
    전체 이미지에 gaussian noise 추가
    """
    noise = rng.normal(0, sigma, size=arr.shape)
    out = arr.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def DropArea(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:
    """
    DropArea 변조

    예시 이미지 스타일 기준:
    - lv1: 색 왜곡 + 작은 DropBlock
    - lv2: Dropout + 강한 노이즈
    - lv3: 더 강한 Dropout + 더 강한 노이즈/색 왜곡
    - lv4: 색 왜곡 + 큰 DropBlock 여러 개
    """

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    np_rng = np.random.default_rng(seed)
    arr = np.array(image.convert("RGB"))
    
    if level == 1:
        # small DropBlock
        arr = _false_color_shift(arr, mode=1)
        arr = _add_dropblock(
            arr,
            block_ratio=0.08,
            num_blocks=1,
            rng=np_rng
        )

    elif level == 2:
        # large DropBlock
        arr = _false_color_shift(arr, mode=4)
        arr = _add_dropblock(
            arr,
            block_ratio=0.22,
            num_blocks=2,
            rng=np_rng
        )

    elif level == 3:
        # medium Dropout
        arr = _false_color_shift(arr, mode=2)
        arr = _add_gaussian_noise(
            arr,
            sigma=38,
            rng=np_rng
        )
        arr = _add_salt_pepper_dropout(
            arr,
            drop_ratio=0.12,
            rng=np_rng
        )

    else:
        # strong Dropout
        arr = _false_color_shift(arr, mode=3)
        arr = _add_gaussian_noise(
            arr,
            sigma=55,
            rng=np_rng
        )
        arr = _add_salt_pepper_dropout(
            arr,
            drop_ratio=0.20,
            rng=np_rng
        )    
    
    
    return Image.fromarray(arr)