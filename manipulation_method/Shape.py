import math
from PIL import Image, ImageDraw


def Shape(image: Image.Image, level: int, aux_image: Image.Image = None, **kwargs) -> Image.Image:
    """
    Shape 변조

    level 1~4에 따라 지정된 도형(circle, polygon, star, heart) 형태 모양으로 
    이미지 영역을 crop한 뒤 원래 이미지 크기로 resize한다.

    level 1: circle (원형 크롭)
    level 2: polygon (다각형/육각형 크롭)
    level 3: star (별 모양 크롭)
    level 4: heart (하트 모양 크롭)
    """

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size
    cx, cy = width / 2.0, height / 2.0
    radius = min(width, height) / 2.0

    # 투명도 또는 마스킹을 위한 알파 마스크 이미지 생성
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    if level == 1:
        # Circle (원)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=255)

    elif level == 2:
        # Polygon (정육각형)
        points = []
        for i in range(6):
            angle = math.radians(60 * i)
            px = cx + radius * math.cos(angle)
            py = cy + radius * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=255)

    elif level == 3:
        # Star (별)
        points = []
        r_outer = radius
        r_inner = radius * 0.4
        for i in range(10):
            r = r_outer if i % 2 == 0 else r_inner
            angle = math.radians(36 * i - 90)
            px = cx + r * math.cos(angle)
            py = cy + r * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=255)

    elif level == 4:
        # Heart (하트)
        points = []
        scale = radius / 16.0
        for t_deg in range(0, 360, 5):
            t = math.radians(t_deg)
            # Parametric equation for heart
            x = 16 * (math.sin(t) ** 3)
            y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
            px = cx + x * scale
            py = cy + y * scale
            points.append((px, py))
        draw.polygon(points, fill=255)

    # 마스크 외곽을 검은색으로 처리한 결과 이미지 생성
    black_bg = Image.new(image.mode, (width, height), 0)
    cropped_shape = Image.composite(image, black_bg, mask)

    shape = cropped_shape.resize(
        (width, height),
        Image.Resampling.LANCZOS
    )

    return shape