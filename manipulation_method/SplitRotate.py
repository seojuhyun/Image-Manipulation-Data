import math
import random
import zlib

from PIL import Image

"""
SplitRotate 변조 (랜덤 분할 + 조각별 랜덤 회전)

이미지를 여러 조각으로 나눈 뒤, 각 조각을 서로 다른 랜덤 각도로
회전시켜 원래 자리에 다시 붙인다.

- 2분할: 좌우 또는 상하로 나누며, 방향은 랜덤이다.
- 4분할: 가로 2 x 세로 2 격자로 나눈다.

각 조각은 36~324도 범위에서 랜덤한 각도만큼 반시계 방향으로 회전한다.
회전으로 생기는 빈 모서리가 보이지 않도록, 회전된 조각 안에서
빈 영역 없이 잘라낼 수 있는 가장 큰 직사각형을 잘라낸 뒤
조각의 원래 크기로 resize한다.
따라서 결과에는 검은 여백이 없고, 조각마다 내용이 확대되어 보인다.
출력 크기는 원본과 같다.

level별 분할 수:
level 1: 2분할
level 2: 2분할
level 3: 4분할
level 4: 4분할

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 분할 수
SPLIT_COUNTS = {
    1: 2,
    2: 2,
    3: 4,
    4: 4,
}

# 조각별 회전 각도 범위 (도)
ANGLE_RANGE = (36, 324)


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


def rotate_piece(piece, angle):
    # 조각을 회전시킨 뒤 빈 영역 없이 잘라내고, 조각의 원래 크기로 되돌린다.

    width, height = piece.size

    rotated = piece.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True
    )

    inner_w, inner_h = max_inner_rect(width, height, angle)

    crop_width = max(1, int(inner_w))
    crop_height = max(1, int(inner_h))

    rotated_width, rotated_height = rotated.size

    left = (rotated_width - crop_width) // 2
    top = (rotated_height - crop_height) // 2

    cropped = rotated.crop(
        (left, top, left + crop_width, top + crop_height)
    )

    return cropped.resize(
        (width, height),
        Image.Resampling.LANCZOS
    )


def SplitRotate(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in SPLIT_COUNTS:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # 분할 방식 결정 (열 수, 행 수)
    if SPLIT_COUNTS[level] == 2:
        # 좌우 또는 상하 분할을 랜덤으로 선택
        cols, rows = (2, 1) if rng.random() < 0.5 else (1, 2)
    else:
        cols, rows = 2, 2

    x_edges = [round(i * width / cols) for i in range(cols + 1)]
    y_edges = [round(j * height / rows) for j in range(rows + 1)]

    SplitRotate = Image.new(image.mode, (width, height))

    for j in range(rows):
        for i in range(cols):

            left, right = x_edges[i], x_edges[i + 1]
            top, bottom = y_edges[j], y_edges[j + 1]

            if right <= left or bottom <= top:
                continue

            piece = image.crop((left, top, right, bottom))

            # 조각마다 회전 각도를 랜덤으로 결정
            angle = rng.uniform(*ANGLE_RANGE)

            SplitRotate.paste(
                rotate_piece(piece, angle),
                (left, top)
            )

    return SplitRotate