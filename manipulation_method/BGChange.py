# level에 따라 background mask 기준을 바꾸지 않고, 동일한 background 영역을 대상으로 새 배경이 섞이는 비율만 25% → 100%
# 이미지의 background를 무작위로 변경. "requires_aux": True
import numpy as np

from PIL import Image, ImageOps


def _estimate_background_mask(
    image: Image.Image
) -> np.ndarray:
    """
    이미지의 테두리 영역을 이용하여
    background 색상 특성을 추정하고,
    background mask를 생성한다.

    반환값:
        background 영역 = True
        foreground 영역 = False
    """

    rgb = np.array(
        image.convert("RGB")
    ).astype(np.float32)

    height, width, _ = rgb.shape


    # ========================================================
    # 이미지 테두리 픽셀 추출
    # ========================================================

    border_size = max(
        1,
        int(
            min(width, height) * 0.05
        )
    )


    top = rgb[
        :border_size,
        :
    ].reshape(-1, 3)

    bottom = rgb[
        height - border_size:,
        :
    ].reshape(-1, 3)

    left = rgb[
        :,
        :border_size
    ].reshape(-1, 3)

    right = rgb[
        :,
        width - border_size:
    ].reshape(-1, 3)


    border_pixels = np.concatenate(
        [
            top,
            bottom,
            left,
            right
        ],
        axis=0
    )


    # ========================================================
    # 대표 background 색상
    # ========================================================

    background_color = np.median(
        border_pixels,
        axis=0
    )


    # ========================================================
    # 색상 거리 계산
    # ========================================================

    border_distances = np.linalg.norm(
        border_pixels
        - background_color,
        axis=1
    )


    image_distances = np.linalg.norm(
        rgb
        - background_color,
        axis=2
    )


    # ========================================================
    # 이미지별 adaptive threshold
    # ========================================================

    threshold = (
        np.percentile(
            border_distances,
            75
        )
        + 15.0
    )


    background_mask = (
        image_distances
        <= threshold
    )


    return background_mask


def _prepare_background(
    aux_image: Image.Image,
    target_size
) -> Image.Image:
    """
    aux image를 target image 크기에 맞게 조정한다.

    비율을 유지하면서 resize 후 center crop한다.
    """

    background = ImageOps.fit(
        aux_image.convert("RGB"),
        target_size,
        method=Image.Resampling.LANCZOS
    )

    return background


def BGChange(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None
) -> Image.Image:
    """
    BGChange 변조

    Target image의 background를 aux image의
    background로 변경한다.

    동일한 background mask와 동일한 aux image를 사용하고,
    level에 따라 새 background의 반영 비율만 증가시킨다.

    level 1: 25%
    level 2: 50%
    level 3: 75%
    level 4: 100%
    """


    # ========================================================
    # 입력 확인
    # ========================================================

    if level not in {
        1,
        2,
        3,
        4
    }:

        raise ValueError(
            "level은 1~4 중 하나여야 합니다."
        )


    if aux_image is None:

        raise ValueError(
            "BGChange는 aux_image가 필요합니다."
        )


    # ========================================================
    # Level별 background change 강도
    # ========================================================

    alpha_map = {
        1: 0.25,
        2: 0.50,
        3: 0.75,
        4: 1.00,
    }

    alpha = alpha_map[
        level
    ]


    # ========================================================
    # Target 이미지 준비
    # ========================================================

    target = image.convert(
        "RGB"
    )

    target_array = np.array(
        target
    ).astype(np.float32)


    # ========================================================
    # Background mask 생성
    # ========================================================

    background_mask = (
        _estimate_background_mask(
            target
        )
    )


    # ========================================================
    # Aux background 준비
    # ========================================================

    new_background = (
        _prepare_background(
            aux_image,
            target.size
        )
    )

    background_array = np.array(
        new_background
    ).astype(np.float32)


    # ========================================================
    # Background blending
    # ========================================================

    result_array = (
        target_array.copy()
    )


    blended_background = (
        (1.0 - alpha)
        * target_array
        + alpha
        * background_array
    )


    result_array[
        background_mask
    ] = blended_background[
        background_mask
    ]


    # ========================================================
    # PIL Image 반환
    # ========================================================

    result_array = np.clip(
        result_array,
        0,
        255
    ).astype(np.uint8)


    result = Image.fromarray(
        result_array
    )


    return result
