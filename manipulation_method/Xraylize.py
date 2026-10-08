# 이미지를 grayscale negative 형태로 변환하여 X-ray와 유사한 효과를 생성하는 코드

from PIL import Image, ImageOps, ImageEnhance


def Xraylize(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    image = image.convert("RGB")

    gray = ImageOps.grayscale(image)
    negative = ImageOps.invert(gray)

    contrast = {
        1: 1.0,
        2: 1.50,
        3: 2.0,
        4: 2.50,
    }[level]

    negative = ImageEnhance.Contrast(
        negative
    ).enhance(contrast)

    return negative.convert("RGB")