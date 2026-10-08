# 비, 안개, 구름, 서리, 눈 중 하나의 날씨 효과를 이미지에 적용하는 코드

from PIL import Image, ImageDraw, ImageFilter
import random
import numpy as np
from manipulation_method.helper_overlay import (_get_default_font, _random_color, _random_pastel_color, _alpha_composite, _random_string, _draw_centered_text, _ensure_rgba)


def Weather(
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

    if level == 1:
        # rain
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for _ in range(int(w * h / 3000)):
            x = rng.randint(0, w)
            y = rng.randint(0, h)
            l = rng.randint(8, 28)
            draw.line((x, y, x + 3, y + l), fill=(220, 220, 255, 90), width=1)
        overlay = overlay.filter(ImageFilter.GaussianBlur(1))
        return _alpha_composite(image, overlay)

    elif level == 2:
        # fog
        overlay = Image.new("RGBA", (w, h), (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        for _ in range(12):
            x = rng.randint(0, w)
            y = rng.randint(0, h)
            r = rng.randint(int(min(w, h) * 0.15), int(min(w, h) * 0.35))
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 255, 255, 70))
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=35))
        return _alpha_composite(image, overlay)

    elif level == 3:
        # clouds
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for _ in range(15):
            x = rng.randint(0, w)
            y = rng.randint(0, int(h * 0.45))
            rw = rng.randint(int(w * 0.08), int(w * 0.18))
            rh = rng.randint(int(h * 0.04), int(h * 0.10))
            draw.ellipse([x-rw, y-rh, x+rw, y+rh], fill=(230, 230, 230, 85))
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=22))
        return _alpha_composite(image, overlay)

    else:
        # snow / frost style
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # snow dots
        for _ in range(int(w * h / 3000)):
            x = rng.randint(0, w)
            y = rng.randint(0, h)
            r = rng.randint(2, 6)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(255, 255, 255, 180))

        # frost edge
        for _ in range(100):
            x = rng.randint(0, w)
            y = rng.randint(0, h)
            if rng.random() < 0.5:
                y = rng.choice([0, h-1])
            else:
                x = rng.choice([0, w-1])

            rr = rng.randint(6, 14)
            draw.ellipse([x-rr, y-rr, x+rr, y+rr], fill=(240, 250, 255, 120))

        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=3))
        return _alpha_composite(image, overlay)