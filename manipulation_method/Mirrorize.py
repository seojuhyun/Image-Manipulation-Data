from PIL import Image


def Mirrorize(image: Image.Image, level: int, aux_image: Image.Image = None, **kwargs) -> Image.Image:
    """
    Mirrorize 변조

    level 1~4에 따라 대칭 반사(Mirror) 방향을 다르게 적용한 뒤
    원래 이미지 크기로 resize한다.

    level 1: 좌측 영역을 우측으로 거울 반사 (좌우 대칭 - 좌 기준)
    level 2: 우측 영역을 좌측으로 거울 반사 (좌우 대칭 - 우 기준)
    level 3: 상단 영역을 하단으로 거울 반사 (상하 대칭 - 상 기준)
    level 4: 하단 영역을 상단으로 거울 반사 (상하 대칭 - 하 기준)
    """

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size
    mirrored = image.copy()

    if level == 1:
        # 좌측 절반을 자르고 좌우 반전 후 우측에 붙이기
        left_half = image.crop((0, 0, width // 2, height))
        flipped = left_half.transpose(Image.FLIP_LEFT_RIGHT)
        mirrored.paste(flipped, (width // 2, 0))

    elif level == 2:
        # 우측 절반을 자르고 좌우 반전 후 좌측에 붙이기
        right_half = image.crop((width // 2, 0, width, height))
        flipped = right_half.transpose(Image.FLIP_LEFT_RIGHT)
        mirrored.paste(flipped, (0, 0))

    elif level == 3:
        # 상단 절반을 자르고 상하 반전 후 하단에 붙이기
        top_half = image.crop((0, 0, width, height // 2))
        flipped = top_half.transpose(Image.FLIP_TOP_BOTTOM)
        mirrored.paste(flipped, (0, height // 2))

    elif level == 4:
        # 하단 절반을 자르고 상하 반전 후 상단에 붙이기
        bottom_half = image.crop((0, height // 2, width, height))
        flipped = bottom_half.transpose(Image.FLIP_TOP_BOTTOM)
        mirrored.paste(flipped, (0, 0))

    mirrorize = mirrored.resize(
        (width, height),
        Image.Resampling.LANCZOS
    )

    return mirrorize