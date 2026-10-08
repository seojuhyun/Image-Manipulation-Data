# 이미지 위에 emoji 또는 이모티콘 문자열을 랜덤 위치에 overlay하는 코드

from PIL import Image, ImageDraw
import random
from manipulation_method.helper_overlay import (_get_default_font, _random_color, _random_pastel_color, _alpha_composite, _random_string, _draw_centered_text, _ensure_rgba)


def OverlayEmoji(
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

    emojis = ["😀", "😂", "😍", "🔥", "⭐", "💥", "❤️", "👍", "🎉", "😎"]
    count = {
        1: 2,
        2: 4,
        3: 6,
        4: 8,
    }[level]

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for _ in range(count):
        txt = rng.choice(emojis)
        font_size = rng.randint(max(18, w // 18), max(28, w // 9))
        font = _get_default_font(font_size)
        x = rng.randint(0, max(0, w - font_size))
        y = rng.randint(0, max(0, h - font_size))
        color = _random_color(rng, alpha=rng.randint(130, 255))
        draw.text((x, y), txt, font=font, fill=color)

    return _alpha_composite(image, overlay)

