# 원본 이미지를 level별로 서로 다른 UI 복잡도를 가진 앱/웹 화면 속 screenshot처럼 합성하는 코드

from PIL import Image, ImageDraw, ImageFont
import random
from manipulation_method.helper_overlay import (_get_default_font, _random_color, _random_pastel_color, _alpha_composite, _random_string, _draw_centered_text, _ensure_rgba)

def _get_default_font(size=24):
    """
    사용 가능한 기본 폰트를 반환한다.
    """
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _fit_image_keep_ratio(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    비율을 유지하면서 target 영역 안에 맞춘다.
    """
    img = img.convert("RGB")
    w, h = img.size

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return canvas


def _draw_icon_row(draw, y, width, labels, font, color=(90, 90, 90)):
    """
    하단 아이콘 행을 그린다.
    """
    n = len(labels)
    for i, label in enumerate(labels):
        x = int((i + 0.5) * width / n)
        draw.text((x - 8, y), label, font=font, fill=color)


def Screenshot(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:
    """
    Screenshot 변조

    원본 이미지를 random한 앱/웹 UI 프레임 안에 삽입하여
    screenshot처럼 보이게 만든다.

    level 기준:
    - lv1: 단순 모바일 피드
    - lv2: 모바일 SNS 포스트
    - lv3: 모바일 SNS + stories + 반응 UI
    - lv4: 데스크탑 웹/SNS 레이아웃
    """

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    rng = random.Random(seed)

    image = image.convert("RGB")
    w, h = image.size

    # --------------------------------------------------------
    # 공통 폰트
    # --------------------------------------------------------
    font_title = _get_default_font(max(18, w // 28))
    font_text = _get_default_font(max(14, w // 42))
    font_small = _get_default_font(max(12, w // 50))

    # ========================================================
    # level 1: 단순 모바일 피드
    # ========================================================
    if level == 1:
        canvas = Image.new("RGB", (w, h), (248, 248, 248))
        draw = ImageDraw.Draw(canvas)

        top_h = int(h * 0.10)
        bottom_h = int(h * 0.08)

        # top bar
        draw.rectangle([0, 0, w, top_h], fill=(252, 252, 252))
        draw.text((18, 12), "network", font=font_title, fill=(25, 25, 25))
        draw.text((w - 35, 12), "⌕", font=font_title, fill=(90, 90, 90))

        # main post frame
        margin = int(w * 0.08)
        card_x0 = margin
        card_x1 = w - margin
        card_y0 = top_h + 18
        card_y1 = h - bottom_h - 20

        draw.rounded_rectangle(
            [card_x0, card_y0, card_x1, card_y1],
            radius=16,
            fill=(255, 255, 255),
            outline=(220, 220, 220)
        )

        # image area 크게
        img_margin = 14
        inner_x0 = card_x0 + img_margin
        inner_x1 = card_x1 - img_margin
        inner_y0 = card_y0 + 16
        inner_y1 = card_y1 - 16

        fitted = _fit_image_keep_ratio(
            image,
            inner_x1 - inner_x0,
            inner_y1 - inner_y0
        )
        canvas.paste(fitted, (inner_x0, inner_y0))

        # bottom nav
        nav_y = h - bottom_h + 10
        _draw_icon_row(
            draw,
            nav_y,
            w,
            ["○", "□", "△", "≡"],
            font_small
        )

        return canvas

    # ========================================================
    # level 2: 모바일 SNS 포스트
    # ========================================================
    elif level == 2:
        canvas = Image.new("RGB", (w, h), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)

        top_h = int(h * 0.09)
        bottom_h = int(h * 0.08)

        draw.rectangle([0, 0, w, top_h], fill=(250, 250, 250))
        draw.text((18, 10), "social", font=font_title, fill=(30, 30, 30))

        margin = int(w * 0.07)
        card_x0 = margin
        card_x1 = w - margin
        card_y0 = top_h + 16
        card_y1 = h - bottom_h - 16

        draw.rounded_rectangle(
            [card_x0, card_y0, card_x1, card_y1],
            radius=16,
            fill=(255, 255, 255),
            outline=(220, 220, 220)
        )

        # post header
        avatar_r = 16
        ax = card_x0 + 16
        ay = card_y0 + 14

        draw.ellipse(
            [ax, ay, ax + 2 * avatar_r, ay + 2 * avatar_r],
            fill=(80, 110, 145)
        )
        draw.text((ax + 42, ay + 3), "John Doe", font=font_text, fill=(35, 35, 35))
        draw.text((ax + 42, ay + 23), "@sampleuser", font=font_small, fill=(120, 120, 120))

        # image area
        img_x0 = card_x0 + 14
        img_x1 = card_x1 - 14
        img_y0 = card_y0 + 60
        img_y1 = card_y1 - 60

        fitted = _fit_image_keep_ratio(
            image,
            img_x1 - img_x0,
            img_y1 - img_y0
        )
        canvas.paste(fitted, (img_x0, img_y0))

        # reaction / caption
        react_y = img_y1 + 10
        draw.text((img_x0, react_y), "♡   ⤴   💬", font=font_text, fill=(70, 70, 70))
        draw.text((img_x0, react_y + 24), "A beautiful day in the city.", font=font_small, fill=(90, 90, 90))

        # bottom nav
        nav_y = h - bottom_h + 10
        _draw_icon_row(
            draw,
            nav_y,
            w,
            ["⌂", "⌕", "＋", "♡", "☺"],
            font_small
        )

        return canvas

    # ========================================================
    # level 3: 모바일 SNS + stories + 반응 UI
    # ========================================================
    elif level == 3:
        canvas = Image.new("RGB", (w, h), (246, 246, 246))
        draw = ImageDraw.Draw(canvas)

        top_h = int(h * 0.08)
        stories_h = int(h * 0.14)
        bottom_h = int(h * 0.09)

        draw.rectangle([0, 0, w, top_h], fill=(252, 252, 252))
        draw.text((18, 8), "network", font=font_title, fill=(25, 25, 25))
        draw.text((w - 36, 8), "⌕", font=font_title, fill=(100, 100, 100))

        # stories bar
        draw.rectangle([0, top_h, w, top_h + stories_h], fill=(250, 250, 250))
        story_y = top_h + 16
        story_r = max(16, w // 22)

        for i in range(6):
            x = 18 + i * int(w * 0.14)
            draw.ellipse(
                [x, story_y, x + 2 * story_r, story_y + 2 * story_r],
                outline=(180, 70, 90),
                width=3
            )
            draw.ellipse(
                [x + 4, story_y + 4, x + 2 * story_r - 4, story_y + 2 * story_r - 4],
                fill=(220, 220, 220)
            )

        margin = int(w * 0.06)
        card_x0 = margin
        card_x1 = w - margin
        card_y0 = top_h + stories_h + 12
        card_y1 = h - bottom_h - 14

        draw.rounded_rectangle(
            [card_x0, card_y0, card_x1, card_y1],
            radius=16,
            fill=(255, 255, 255),
            outline=(220, 220, 220)
        )

        # post header
        avatar_r = 16
        ax = card_x0 + 16
        ay = card_y0 + 14
        draw.ellipse([ax, ay, ax + 2 * avatar_r, ay + 2 * avatar_r], fill=(70, 100, 130))
        draw.text((ax + 42, ay + 2), "John Doe", font=font_text, fill=(40, 40, 40))
        draw.text((card_x1 - 30, ay + 2), "⋯", font=font_title, fill=(90, 90, 90))

        # image area
        img_x0 = card_x0 + 14
        img_x1 = card_x1 - 14
        img_y0 = card_y0 + 54
        img_y1 = card_y1 - 82

        fitted = _fit_image_keep_ratio(
            image,
            img_x1 - img_x0,
            img_y1 - img_y0
        )
        canvas.paste(fitted, (img_x0, img_y0))

        # reactions
        react_y = img_y1 + 10
        draw.text((img_x0, react_y), "♡   💬   ↗   🔖", font=font_text, fill=(70, 70, 70))
        draw.text((img_x0, react_y + 24), "Liked by user_01 and 124 others", font=font_small, fill=(60, 60, 60))
        draw.text((img_x0, react_y + 44), "View all 8 comments", font=font_small, fill=(120, 120, 120))

        # bottom nav
        nav_y = h - bottom_h + 10
        _draw_icon_row(
            draw,
            nav_y,
            w,
            ["⌂", "▶", "＋", "♡", "☺"],
            font_small
        )

        return canvas

    # ========================================================
    # level 4: 데스크탑 웹/SNS 레이아웃
    # ========================================================
    else:
        canvas = Image.new("RGB", (w, h), (242, 244, 247))
        draw = ImageDraw.Draw(canvas)

        top_h = int(h * 0.08)
        draw.rectangle([0, 0, w, top_h], fill=(252, 252, 252))
        draw.text((20, 10), "network", font=font_title, fill=(20, 20, 20))
        draw.text((w - 120, 10), "Home   Feed", font=font_small, fill=(90, 90, 90))
        draw.text((w - 36, 10), "⌕", font=font_title, fill=(90, 90, 90))

        # layout
        left_w = int(w * 0.18)
        right_w = int(w * 0.22)
        center_x0 = left_w + 14
        center_x1 = w - right_w - 14
        body_y0 = top_h + 12
        body_y1 = h - 12

        # left sidebar
        draw.rounded_rectangle(
            [12, body_y0, left_w, body_y1],
            radius=12,
            fill=(255, 255, 255),
            outline=(225, 225, 225)
        )

        menu_items = ["Profile", "Explore", "Messages", "Favorites", "Settings"]
        for i, item in enumerate(menu_items):
            draw.text((22, body_y0 + 20 + i * 28), item, font=font_small, fill=(80, 80, 80))

        # right sidebar
        draw.rounded_rectangle(
            [w - right_w, body_y0, w - 12, body_y1],
            radius=12,
            fill=(255, 255, 255),
            outline=(225, 225, 225)
        )

        draw.text((w - right_w + 12, body_y0 + 18), "Contacts", font=font_text, fill=(50, 50, 50))
        for i in range(6):
            cy = body_y0 + 52 + i * 34
            draw.ellipse([w - right_w + 12, cy, w - right_w + 32, cy + 20], fill=(150, 160, 180))
            draw.text((w - right_w + 40, cy + 2), f"user_{i+1}", font=font_small, fill=(90, 90, 90))

        # center post
        draw.rounded_rectangle(
            [center_x0, body_y0, center_x1, body_y1],
            radius=14,
            fill=(255, 255, 255),
            outline=(220, 220, 220)
        )

        # post header
        avatar_r = 16
        ax = center_x0 + 16
        ay = body_y0 + 14
        draw.ellipse([ax, ay, ax + 2 * avatar_r, ay + 2 * avatar_r], fill=(70, 100, 130))
        draw.text((ax + 42, ay + 2), "John Doe", font=font_text, fill=(35, 35, 35))
        draw.text((ax + 42, ay + 22), "2 hrs ago", font=font_small, fill=(130, 130, 130))

        # image area
        img_x0 = center_x0 + 14
        img_x1 = center_x1 - 14
        img_y0 = body_y0 + 60
        img_y1 = body_y1 - 120

        fitted = _fit_image_keep_ratio(
            image,
            img_x1 - img_x0,
            img_y1 - img_y0
        )
        canvas.paste(fitted, (img_x0, img_y0))

        # footer info
        info_y = img_y1 + 12
        draw.text((img_x0, info_y), "♡  2.1k    💬  184    ↗  Share", font=font_text, fill=(70, 70, 70))
        draw.text((img_x0, info_y + 28), "This post was shared in your feed.", font=font_small, fill=(100, 100, 100))

        # comments area
        comment_y = info_y + 56
        for i in range(2):
            yy = comment_y + i * 26
            draw.ellipse([img_x0, yy, img_x0 + 16, yy + 16], fill=(170, 180, 190))
            draw.text((img_x0 + 24, yy), f"user_{i+1}: Nice shot!", font=font_small, fill=(90, 90, 90))

        return canvas