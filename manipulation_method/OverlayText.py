# 이미지 위에 랜덤 문자열 또는 짧은 문구를 반투명하게 overlay하는 코드

from PIL import Image, ImageDraw
import random
from manipulation_method.helper_overlay import (_get_default_font, _random_color, _random_pastel_color, _alpha_composite, _random_string, _draw_centered_text, _ensure_rgba)


def OverlayText(
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

    count = {
        1: 1,
        2: 2,
        3: 4,
        4: 6,
    }[level]

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    for _ in range(count):
        draw = ImageDraw.Draw(overlay)
        text = _random_string(rng, 4, 10)
        font_size = rng.randint(max(18, w // 22), max(30, w // 10))
        font = _get_default_font(font_size)

        x = rng.randint(0, max(0, w - 100))
        y = rng.randint(0, max(0, h - 40))
        fill = _random_color(rng, alpha=rng.randint(80, 180))

        txt_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        txt_draw = ImageDraw.Draw(txt_layer)
        txt_draw.text((x, y), text, font=font, fill=fill)
        txt_layer = txt_layer.rotate(rng.uniform(-20, 20), resample=Image.Resampling.BICUBIC)

        overlay = Image.alpha_composite(overlay, txt_layer)

    return _alpha_composite(image, overlay)

