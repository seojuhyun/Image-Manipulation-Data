# 보조 이미지(aux_image) 또는 원본 crop을 sticker처럼 축소/회전하여 원본 이미지 위에 overlay하는 코드

from PIL import Image
import random
from manipulation_method.helper_overlay import (_get_default_font, _random_color, _random_pastel_color, _alpha_composite, _random_string, _draw_centered_text, _ensure_rgba)


def OverlayImage(
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

    base = image.convert("RGBA")

    count = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
    }[level]

    for _ in range(count):
        if aux_image is not None:
            sticker = aux_image.convert("RGBA")
        else:
            # 원본 일부를 crop해서 sticker처럼 사용
            crop_w = rng.randint(int(w * 0.18), int(w * 0.35))
            crop_h = rng.randint(int(h * 0.18), int(h * 0.35))
            x0 = rng.randint(0, max(0, w - crop_w))
            y0 = rng.randint(0, max(0, h - crop_h))
            sticker = image.crop((x0, y0, x0 + crop_w, y0 + crop_h)).convert("RGBA")

        target_w = rng.randint(int(w * 0.18), int(w * 0.35))
        target_h = int(sticker.height * target_w / max(1, sticker.width))
        sticker = sticker.resize((target_w, target_h), Image.Resampling.LANCZOS)

        rot = rng.uniform(-20, 20)
        sticker = sticker.rotate(rot, resample=Image.Resampling.BICUBIC, expand=True)

        # opacity 조절
        alpha = sticker.getchannel("A")
        alpha = alpha.point(lambda p: int(p * rng.uniform(0.55, 0.85)))
        sticker.putalpha(alpha)

        x = rng.randint(0, max(0, w - sticker.width))
        y = rng.randint(0, max(0, h - sticker.height))
        base.paste(sticker, (x, y), sticker)

    return base.convert("RGB")
