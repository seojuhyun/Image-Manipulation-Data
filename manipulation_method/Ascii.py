# 원본 이미지를 옅게 유지하고 그 위에 colored ASCII 문자를 배치하여 형태를 더 잘 보이게 하는 코드

from PIL import Image, ImageDraw


ASCII_CHARS = "@#%*+=-:./()!?&"


def Ascii(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    image = image.convert("RGB")

    width, height = image.size

    # level이 높아질수록 문자 크기를 키워/ 기존보다 문자를 작게 해서 디테일을 더 살림
    # ASCII 효과를 더 강하게 적용
    cell_size = {
        1: 4,
        2: 5,
        3: 6,
        4: 8,
    }[level]

    # 원본 이미지를 밝게 만들어 배경으로 사용
    white = Image.new(
        "RGB",
        image.size,
        (255, 255, 255)
    )

    # 원본 형태를 어느 정도 유지/ 이 값이 클수록 원본 이미지가 더 많이 보입니다.
    background_ratio = {
        1: 0.50,
        2: 0.40,
        3: 0.30,
        4: 0.20,
    }[level]

    result = Image.blend(
        white,
        image,
        background_ratio
    )

    draw = ImageDraw.Draw(result)

    for y in range(
        0,
        height,
        cell_size
    ):
        for x in range(
            0,
            width,
            cell_size
        ):

            r, g, b = image.getpixel(
                (
                    min(x, width - 1),
                    min(y, height - 1)
                )
            )

            brightness = (
                0.299 * r
                + 0.587 * g
                + 0.114 * b
            )

            idx = int(
                (255 - brightness)
                / 255
                * (len(ASCII_CHARS) - 1)
            )

            char = ASCII_CHARS[idx]

            # 기존보다 문자 색을 더 진하게 유지
            color = (
                int(r * 0.75),
                int(g * 0.75),
                int(b * 0.75),
            )

            draw.text(
                (x, y),
                char,
                fill=color
            )

    return result