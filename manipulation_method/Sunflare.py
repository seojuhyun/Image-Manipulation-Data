# 이미지 위에 밝은 광원과 lens flare 원형 halo를 추가하는 코드

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import random
import math
import random
import math
from manipulation_method.helper_overlay import (_get_default_font, _random_color, _random_pastel_color, _alpha_composite, _random_string, _draw_centered_text, _ensure_rgba)


def _alpha_composite(base: Image.Image, overlay: Image.Image) -> Image.Image:
    base_rgba = base.convert("RGBA")
    overlay_rgba = overlay.convert("RGBA")
    result = Image.alpha_composite(base_rgba, overlay_rgba)
    return result.convert("RGB")


def Sunflare(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:
    """
    Sunflare 변조

    fog처럼 뿌옇게 만드는 대신,
    밝은 태양광 + 노란빛 glow + lens flare를 추가하여
    화사한 느낌을 만든다.

    level 기준:
    - lv1: 약한 sunflare
    - lv2: 중간 sunflare
    - lv3: 강한 sunflare
    - lv4: 매우 강한 sunflare
    """

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    rng = random.Random(seed)
    image = image.convert("RGB")
    w, h = image.size

    # --------------------------------------------------------
    # 원본 전체 밝기/채도 약간 증가
    # --------------------------------------------------------
    bright_factor = {
        1: 1.05,
        2: 1.10,
        3: 1.15,
        4: 1.20,
    }[level]

    color_factor = {
        1: 1.03,
        2: 1.06,
        3: 1.10,
        4: 1.15,
    }[level]

    base = ImageEnhance.Brightness(image).enhance(bright_factor)
    base = ImageEnhance.Color(base).enhance(color_factor)

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # --------------------------------------------------------
    # 태양 위치: 보통 상단 쪽
    # --------------------------------------------------------
    sun_x = rng.randint(int(w * 0.15), int(w * 0.85))
    sun_y = rng.randint(0, int(h * 0.22))

    # --------------------------------------------------------
    # level별 강도
    # --------------------------------------------------------
    glow_scale = {
        1: 0.16,
        2: 0.20,
        3: 0.24,
        4: 0.28,
    }[level]

    core_scale = {
        1: 0.05,
        2: 0.06,
        3: 0.07,
        4: 0.08,
    }[level]

    flare_count = {
        1: 3,
        2: 4,
        3: 6,
        4: 8,
    }[level]

    blur_radius = {
        1: 10,
        2: 14,
        3: 18,
        4: 24,
    }[level]

    # --------------------------------------------------------
    # 1. 큰 노란빛 glow
    # --------------------------------------------------------
    glow_r = int(min(w, h) * glow_scale)
    for scale, color in [
        (1.8, (255, 220, 120, 35)),
        (1.3, (255, 230, 150, 55)),
        (1.0, (255, 245, 200, 70)),
    ]:
        r = int(glow_r * scale)
        draw.ellipse(
            [sun_x - r, sun_y - r, sun_x + r, sun_y + r],
            fill=color
        )

    # --------------------------------------------------------
    # 2. 태양 core (밝은 흰색/노란색 중심)
    # --------------------------------------------------------
    core_r = int(min(w, h) * core_scale)
    for scale, color in [
        (1.8, (255, 250, 220, 120)),
        (1.2, (255, 255, 235, 180)),
        (0.8, (255, 255, 255, 235)),
    ]:
        r = int(core_r * scale)
        draw.ellipse(
            [sun_x - r, sun_y - r, sun_x + r, sun_y + r],
            fill=color
        )

    # --------------------------------------------------------
    # 3. diagonal lens flare dots
    # 태양 반대 방향으로 이어지는 밝은 점들
    # --------------------------------------------------------
    center_x = w / 2
    center_y = h / 2

    dir_x = center_x - sun_x
    dir_y = center_y - sun_y

    for i in range(1, flare_count + 1):
        t = i / (flare_count + 1)

        fx = int(sun_x + dir_x * t * 1.8)
        fy = int(sun_y + dir_y * t * 1.8)

        r = int(min(w, h) * (0.01 + (flare_count - i + 1) * 0.008))
        alpha = max(35, 110 - i * 10)

        flare_color = rng.choice([
            (255, 240, 180, alpha),
            (255, 220, 160, alpha),
            (255, 255, 255, alpha),
            (255, 235, 200, alpha),
        ])

        draw.ellipse(
            [fx - r, fy - r, fx + r, fy + r],
            fill=flare_color
        )

    # --------------------------------------------------------
    # 4. 약한 빛줄기(streak)
    # --------------------------------------------------------
    streak_len = int(min(w, h) * 0.40)
    for angle_deg in [-25, 0, 20]:
        angle = math.radians(angle_deg + rng.uniform(-6, 6))

        dx = math.cos(angle) * streak_len
        dy = math.sin(angle) * streak_len

        draw.line(
            (sun_x - dx, sun_y - dy, sun_x + dx, sun_y + dy),
            fill=(255, 240, 200, 35),
            width=max(2, min(w, h) // 200)
        )

    # --------------------------------------------------------
    # 5. blur로 자연스럽게 번짐
    # --------------------------------------------------------
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    result = _alpha_composite(base, overlay)

    # --------------------------------------------------------
    # 6. 최종적으로 조금 더 화사하게
    # --------------------------------------------------------
    result = ImageEnhance.Brightness(result).enhance(
        {
            1: 1.02,
            2: 1.05,
            3: 1.08,
            4: 1.12,
        }[level]
    )

    return result