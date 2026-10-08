# 이미지 위에 삼각형/다각형 형태의 반투명 그림자를 추가하는 코드

from PIL import Image, ImageDraw, ImageFilter
import random
import math
from manipulation_method.helper_overlay import (_get_default_font, _random_color, _random_pastel_color, _alpha_composite, _random_string, _draw_centered_text, _ensure_rgba)


def Shadow(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    rng = random.Random(seed)
    image = image.convert("RGB")
    w, h = image.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    num_shapes = {
        1: 3,
        2: 6,
        3: 9,
        4: 12,
    }[level]

    for _ in range(num_shapes):
        cx = rng.randint(0, w)
        cy = rng.randint(int(h * 0.3), h)
        radius = rng.randint(int(min(w, h) * 0.12), int(min(w, h) * 0.25))
        sides = rng.randint(3, 6)

        pts = []
        for i in range(sides):
            ang = 2 * math.pi * i / sides + rng.uniform(-0.25, 0.25)
            rr = radius * rng.uniform(0.6, 1.4)
            pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))

        draw.polygon(pts, fill=(0, 0, 0, rng.randint(45, 95)))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=6))
    return _alpha_composite(image, overlay)