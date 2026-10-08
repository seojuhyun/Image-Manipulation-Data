# 이미지 위에 반투명한 색상 stripe를 random 방향으로 여러 개 overlay하는 코드

from PIL import Image, ImageDraw
import random
import math
from manipulation_method.helper_overlay import (_get_default_font, _random_color, _random_pastel_color, _alpha_composite, _random_string, _draw_centered_text, _ensure_rgba)


def AddStripes(
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

    stripe_count = {
        1: 5,
        2: 10,
        3: 15,
        4: 20,
    }[level]

    stripe_width = {
        1: int(min(w, h) * 0.12),
        2: int(min(w, h) * 0.10),
        3: int(min(w, h) * 0.14),
        4: int(min(w, h) * 0.05),
    }[level]

    alpha = {
        1: 70,
        2: 90,
        3: 80,
        4: 120,
    }[level]

    # level 4는 세로 stripe 느낌
    if level == 4:
        base_angle = rng.uniform(-20, 20)
    else:
        base_angle = rng.uniform(20, 160)

    diag = int(math.sqrt(w * w + h * h)) * 2
    center = (w / 2, h / 2)

    for _ in range(stripe_count):
        angle = math.radians(base_angle + rng.uniform(-10, 10))

        offset = rng.randint(-diag // 2, diag // 2)
        cx = center[0] + offset * math.cos(angle + math.pi / 2)
        cy = center[1] + offset * math.sin(angle + math.pi / 2)

        dx = math.cos(angle) * diag
        dy = math.sin(angle) * diag
        px = math.cos(angle + math.pi / 2) * stripe_width / 2
        py = math.sin(angle + math.pi / 2) * stripe_width / 2

        polygon = [
            (cx - dx - px, cy - dy - py),
            (cx + dx - px, cy + dy - py),
            (cx + dx + px, cy + dy + py),
            (cx - dx + px, cy - dy + py),
        ]

        color = (
            rng.randint(80, 255),
            rng.randint(80, 255),
            rng.randint(80, 255),
            alpha
        )
        draw.polygon(polygon, fill=color)

    return _alpha_composite(image, overlay)