# 이미지를 색상 block으로 단순화하고 각 block에 LEGO stud를 그려 LEGO mosaic 형태로 변환하는 코드

from PIL import Image, ImageDraw, ImageEnhance


def Legoify(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    image = image.convert("RGB")

    width, height = image.size

    block_size = {
        1: 8,
        2: 12,
        3: 16,
        4: 20,
    }[level]

    small_w = max(
        1,
        width // block_size
    )

    small_h = max(
        1,
        height // block_size
    )

    small = image.resize(
        (small_w, small_h),
        Image.Resampling.BILINEAR
    )

    result = small.resize(
        (width, height),
        Image.Resampling.NEAREST
    )

    result = ImageEnhance.Color(
        result
    ).enhance(1.15)

    draw = ImageDraw.Draw(result)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):

            cx = x + block_size // 2
            cy = y + block_size // 2

            r = max(
                2,
                int(block_size * 0.28)
            )

            draw.ellipse(
                (
                    cx - r,
                    cy - r,
                    cx + r,
                    cy + r
                ),
                outline=(120, 120, 120),
                width=max(1, block_size // 12)
            )

            # LEGO stud highlight
            r2 = max(1, r // 2)

            draw.arc(
                (
                    cx - r2,
                    cy - r2,
                    cx + r2,
                    cy + r2
                ),
                180,
                300,
                fill=(235, 235, 235)
            )

    return result