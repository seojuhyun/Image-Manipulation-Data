import math
import random
import zlib

from PIL import Image

"""
Rotate 변조 (랜덤 회전)

이미지를 level별 각도 범위 안에서 랜덤한 각도만큼
반시계 방향으로 회전시킨다.
회전으로 생기는 빈 모서리가 보이지 않도록, 회전된 이미지 안에서
빈 영역 없이 잘라낼 수 있는 가장 큰 직사각형을 중앙에서 잘라낸 뒤
원래 이미지 크기로 resize한다.
따라서 결과에는 검은 여백이 없고, 회전 각도에 따라 내용이 확대되어 보인다.

level별 회전 각도 범위 (72도 간격 중심값 ± 36도):
level 1: 36~108도
level 2: 108~180도
level 3: 180~252도
level 4: 252~324도

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 회전 각도 범위 (도)
ANGLE_RANGES = {
    1: (36, 108),
    2: (108, 180),
    3: (180, 252),
    4: (252, 324),
}


def max_inner_rect(width, height, angle_deg):
    # width x height 이미지를 angle_deg만큼 회전했을 때,
    # 빈 영역 없이 들어가는 가장 큰 직사각형의 (가로, 세로)를 구한다.

    angle = math.radians(angle_deg)
    sin_a = abs(math.sin(angle))
    cos_a = abs(math.cos(angle))

    width_is_longer = width >= height
    side_long, side_short = (
        (width, height) if width_is_longer else (height, width)
    )

    if (
        side_short <= 2.0 * sin_a * cos_a * side_long
        or abs(sin_a - cos_a) < 1e-10
    ):
        # 짧은 변이 제약이 되는 경우
        x = 0.5 * side_short
        if width_is_longer:
            inner_w, inner_h = x / sin_a, x / cos_a
        else:
            inner_w, inner_h = x / cos_a, x / sin_a
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        inner_w = (width * cos_a - height * sin_a) / cos_2a
        inner_h = (height * cos_a - width * sin_a) / cos_2a

    return inner_w, inner_h


def Rotate(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in ANGLE_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # level별 범위 안에서 회전 각도를 랜덤으로 결정
    angle = rng.uniform(*ANGLE_RANGES[level])

    # 잘림 없이 회전 (캔버스 확장)
    rotated = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True
    )

    # 빈 영역이 없는 가장 큰 직사각형을 중앙에서 잘라낸다.
    inner_w, inner_h = max_inner_rect(width, height, angle)

    crop_width = max(1, int(inner_w))
    crop_height = max(1, int(inner_h))

    rotated_width, rotated_height = rotated.size

    left = (rotated_width - crop_width) // 2
    top = (rotated_height - crop_height) // 2

    cropped = rotated.crop(
        (left, top, left + crop_width, top + crop_height)
    )

    # 원래 크기로 복원
    resized = cropped.resize(
        (width, height),
        Image.Resampling.LANCZOS
    )

    return resized