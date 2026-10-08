# 여러 overlay/augmentation 기법에서 공통으로 사용하는 helper 함수들

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import cv2
import random
import math
import string


def _get_default_font(size=32):
    """
    사용 가능한 기본 폰트를 반환한다.
    """
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _random_color(rng, alpha=255):
    """
    random RGBA 색상 생성
    """
    return (
        rng.randint(0, 255),
        rng.randint(0, 255),
        rng.randint(0, 255),
        alpha
    )


def _random_pastel_color(rng, alpha=180):
    """
    연한 overlay용 pastel 색상 생성
    """
    return (
        rng.randint(120, 255),
        rng.randint(120, 255),
        rng.randint(120, 255),
        alpha
    )


def _alpha_composite(base: Image.Image, overlay: Image.Image) -> Image.Image:
    """
    RGBA overlay를 alpha composite한다.
    """
    base_rgba = base.convert("RGBA")
    overlay_rgba = overlay.convert("RGBA")
    result = Image.alpha_composite(base_rgba, overlay_rgba)
    return result.convert("RGB")


def _random_string(rng, min_len=4, max_len=9):
    """
    랜덤 문자열 생성
    """
    length = rng.randint(min_len, max_len)
    chars = string.ascii_letters
    return "".join(rng.choice(chars) for _ in range(length))


def _draw_centered_text(draw, box, text, font, fill):
    """
    box 중앙에 텍스트를 그린다.
    """
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = x0 + (x1 - x0 - tw) / 2
    y = y0 + (y1 - y0 - th) / 2
    draw.text((x, y), text, font=font, fill=fill)


def _ensure_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA")