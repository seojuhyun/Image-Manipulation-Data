# ellipse, rectangle, triangle, star, pentagon 등의 도형을 반투명하게 여러 개 overlay하는 코드

from PIL import Image, ImageDraw
import random
import math
from manipulation_method.helper_overlay import (_get_default_font, _random_color, _random_pastel_color, _alpha_composite, _random_string, _draw_centered_text, _ensure_rgba)


def AddShapes(
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

    shape_count = {
        1: 30,
        2: 50,
        3: 70,
        4: 90,
    }[level]

    allowed_shapes = {
        1: ["ellipse"],
        2: ["triangle"],
        3: ["star", "ellipse"],
        4: ["star", "ellipse", "triangle"],
    }[level]

    for _ in range(shape_count):
        shape_type = rng.choice(allowed_shapes)
        x = rng.randint(0, w)
        y = rng.randint(0, h)
        size = rng.randint(6, 22) if level != 4 else rng.randint(12, 36)

        color = _random_pastel_color(rng, alpha=rng.randint(90, 160))

        if shape_type == "ellipse":
            draw.ellipse([x-size, y-size, x+size, y+size], fill=color)

        elif shape_type == "rectangle":
            draw.rectangle([x-size, y-size, x+size, y+size], fill=color)

        elif shape_type == "triangle":
            pts = [(x, y-size), (x-size, y+size), (x+size, y+size)]
            draw.polygon(pts, fill=color)

        elif shape_type == "pentagon":
            pts = []
            for i in range(5):
                ang = 2 * math.pi * i / 5 - math.pi / 2
                pts.append((x + size * math.cos(ang), y + size * math.sin(ang)))
            draw.polygon(pts, fill=color)

        elif shape_type == "star":
            pts = []
            for i in range(10):
                ang = 2 * math.pi * i / 10 - math.pi / 2
                rr = size if i % 2 == 0 else size * 0.45
                pts.append((x + rr * math.cos(ang), y + rr * math.sin(ang)))
            draw.polygon(pts, fill=color)

    return _alpha_composite(image, overlay)