# 이미지 전체를 회전/이동/확대축소한 variant들에 풍차 모양 sector mask를 씌워 kaleidoscope 효과를 만드는 코드

from PIL import Image
import numpy as np
import cv2
import random


def _angle_diff(a, b):
    """
    두 각도 차이를 [-pi, pi] 범위로 정규화한다.
    """
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def _make_sector_mask(
    h: int,
    w: int,
    center_x: float,
    center_y: float,
    sector_center_angle: float,
    sector_half_angle: float,
    outer_radius: float
) -> np.ndarray:
    """
    풍차 blade 하나에 해당하는 sector mask를 만든다.
    """
    yy, xx = np.indices((h, w), dtype=np.float32)

    dx = xx - center_x
    dy = yy - center_y

    radius = np.sqrt(dx ** 2 + dy ** 2)
    angle = np.arctan2(dy, dx)

    angle_diff = _angle_diff(angle, sector_center_angle)

    mask = (
        (radius <= outer_radius)
        & (np.abs(angle_diff) <= sector_half_angle)
    )

    return mask


def _transform_full_image(
    arr: np.ndarray,
    angle_deg: float,
    scale: float,
    shift_x: float,
    shift_y: float
) -> np.ndarray:
    """
    이미지 전체를 회전/스케일/이동 변환한다.
    """
    h, w = arr.shape[:2]

    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(
        center,
        angle_deg,
        scale
    )

    M[0, 2] += shift_x
    M[1, 2] += shift_y

    transformed = cv2.warpAffine(
        arr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return transformed


def Kaleidoscope(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:
    """
    Kaleidoscope 변조

    기존처럼 이미지 중심의 픽셀을 반복 사용하는 대신,
    이미지 전체를 회전/이동/스케일한 여러 variant를 만들고
    풍차 모양 blade mask로 잘라 합성한다.

    level 기준:
    - level 1 -> 4 blades
    - level 2 -> 6 blades
    - level 3 -> 8 blades
    - level 4 -> 10 blades

    background_mode:
    - "black" : 검은 배경 위에 kaleidoscope 조각 배치
    - "image" : 원본 이미지 위에 검은 gap이 덮인 형태
    """

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    rng = random.Random(seed)

    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]

    cx = w / 2.0
    cy = h / 2.0

    # ----------------------------------------
    # background 설정
    # "black" 또는 "image"
    # ----------------------------------------
    background_mode = "black"
    # background_mode = "image"

    if background_mode == "black":
        result = np.zeros_like(arr)
    else:
        result = arr.copy()

    # ----------------------------------------
    # blade 개수
    # ----------------------------------------
    num_blades = {
        1: 4,
        2: 5,
        3: 6,
        4: 7,
    }[level]

    # ----------------------------------------
    # blade가 차지하는 각도 비율
    # 작을수록 gap이 넓어짐
    # ----------------------------------------
    fill_ratio = {
        1: 0.8,
        2: 0.8,
        3: 0.8,
        4: 0.8,
    }[level]

    period = 2.0 * np.pi / num_blades
    sector_angle = period * fill_ratio
    sector_half_angle = sector_angle / 2.0

    # ----------------------------------------
    # 바깥 원 반지름
    # ----------------------------------------
    outer_radius = min(cx, cy)

    # ----------------------------------------
    # level이 높을수록 더 다양한 전체 이미지 변환
    # ----------------------------------------
    max_rotation_jitter = {
        1: 8,
        2: 12,
        3: 18,
        4: 24,
    }[level]

    max_shift = {
        1: 10,
        2: 18,
        3: 26,
        4: 34,
    }[level]

    scale_range = {
        1: (0.98, 1.05),
        2: (0.95, 1.10),
        3: (0.92, 1.15),
        4: (0.90, 1.20),
    }[level]

    # ----------------------------------------
    # 각 blade별로
    # "이미지 전체"를 변형한 뒤
    # sector mask를 씌워 합성
    # ----------------------------------------
    for i in range(num_blades):

        sector_center_angle = i * period

        # 각 blade마다 다른 전체 이미지 variant 생성
        angle_deg = (
            (360.0 / num_blades) * i
            + rng.uniform(-max_rotation_jitter, max_rotation_jitter)
        )

        scale = rng.uniform(
            scale_range[0],
            scale_range[1]
        )

        shift_x = rng.uniform(
            -max_shift,
            max_shift
        )

        shift_y = rng.uniform(
            -max_shift,
            max_shift
        )

        transformed = _transform_full_image(
            arr=arr,
            angle_deg=angle_deg,
            scale=scale,
            shift_x=shift_x,
            shift_y=shift_y
        )

        mask = _make_sector_mask(
            h=h,
            w=w,
            center_x=cx,
            center_y=cy,
            sector_center_angle=sector_center_angle,
            sector_half_angle=sector_half_angle,
            outer_radius=outer_radius
        )

        result[mask] = transformed[mask]

    # ----------------------------------------
    # background_mode == "image" 인 경우
    # blade가 아닌 gap 부분을 검게 만들어
    # "원본 이미지에 검은 패턴을 씌운 듯한" 결과를 만듦
    # ----------------------------------------
    if background_mode == "image":
        yy, xx = np.indices((h, w), dtype=np.float32)

        dx = xx - cx
        dy = yy - cy
        radius = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.arctan2(dy, dx)

        combined_mask = np.zeros((h, w), dtype=bool)

        for i in range(num_blades):
            sector_center_angle = i * period
            blade_mask = _make_sector_mask(
                h=h,
                w=w,
                center_x=cx,
                center_y=cy,
                sector_center_angle=sector_center_angle,
                sector_half_angle=sector_half_angle,
                outer_radius=outer_radius
            )
            combined_mask |= blade_mask

        circle_mask = radius <= outer_radius
        gap_mask = circle_mask & (~combined_mask)

        result[gap_mask] = 0

    return Image.fromarray(result)