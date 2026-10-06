from PIL import Image


def HorizFlip(image: Image.Image, level: int, aux_image: Image.Image = None, **kwargs) -> Image.Image:
    """
    HorizFlip 변조

    level 1~4에 상관없이 좌우 반전(Horizontal Flip)을 적용한다.

    level 1: 좌우 반전 적용
    level 2: 좌우 반전 적용
    level 3: 좌우 반전 적용
    level 4: 좌우 반전 적용
    """

    # ========================================================
    # 입력 확인
    # ========================================================

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")
    
    # ========================================================
    # Horizontal Flip 적용
    # ========================================================

    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)

    return flipped