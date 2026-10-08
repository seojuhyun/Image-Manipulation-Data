# 이미지 상단에 컬러 banner와 랜덤 텍스트를 추가하여 meme format처럼 보이게 만드는 코드

from PIL import Image, ImageDraw, ImageFont
import random
from manipulation_method.helper_overlay import (_get_default_font, _random_color, _random_pastel_color, _alpha_composite, _random_string, _draw_centered_text, _ensure_rgba)


def MemeFormat(
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

    banner_h = int(h * 0.18)
    canvas = Image.new("RGB", (w, h + banner_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    banner_colors = [
        (50, 220, 70),
        (170, 240, 230),
        (20, 0, 30),
        (255, 80, 210),
    ]
    banner_color = banner_colors[level - 1]
    draw.rectangle([0, 0, w, banner_h], fill=banner_color)

    canvas.paste(image, (0, banner_h))

    text = _random_string(rng, 4, 8)
    font = _get_default_font(max(28, w // 8))

    text_color = [
        (0, 0, 0),
        (90, 140, 120),
        (80, 40, 255),
        (220, 40, 70),
    ][level - 1]

    # 살짝 rotation 느낌 주기 위해 별도 layer에 그림
    txt_layer = Image.new("RGBA", (w, banner_h), (0, 0, 0, 0))
    txt_draw = ImageDraw.Draw(txt_layer)
    bbox = txt_draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (w - tw) // 2
    ty = (banner_h - th) // 2
    txt_draw.text((tx, ty), text, font=font, fill=text_color + (255,))

    rot = rng.uniform(-8, 8)
    txt_layer = txt_layer.rotate(rot, resample=Image.Resampling.BICUBIC, expand=False)
    canvas = _alpha_composite(canvas, Image.new("RGBA", canvas.size, (0, 0, 0, 0)).copy())
    canvas_rgba = canvas.convert("RGBA")
    temp = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    temp.paste(txt_layer, (0, 0), txt_layer)
    return Image.alpha_composite(canvas_rgba, temp).convert("RGB")