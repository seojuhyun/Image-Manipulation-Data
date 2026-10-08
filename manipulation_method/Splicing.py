# level에 따라 전체 이미지 대비 총 변조 면적 비율을 5%, 10%, 20%, 30%로 설정
# "requires_aux": True,

import math
import random
from PIL import Image


def Splicing(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    seed: int | None = None
) -> Image.Image:
    """
    Splicing 변조

    다른 이미지(aux_image)에서 영역을 가져와
    target image(image)의 임의 위치에 붙여넣는다.

    level은 총 변조 면적 비율 기준으로 정의한다.

    level 1: 약 5%
    level 2: 약 10%
    level 3: 약 20%
    level 4: 약 30%
    """

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    if aux_image is None:
        raise ValueError("Splicing은 aux_image가 필요합니다.")

    rng = random.Random(seed)

    target = image.convert("RGB").copy()
    source = aux_image.convert("RGB")

    tw, th = target.size
    sw, sh = source.size

    # --------------------------------------------------------
    # level별 총 변조 면적 비율
    # --------------------------------------------------------
    area_ratios = {
        1: 0.05,   # 5%
        2: 0.10,   # 10%
        3: 0.20,   # 20%
        4: 0.30,   # 30%
    }

    # --------------------------------------------------------
    # level별 patch 개수
    # --------------------------------------------------------
    num_patches_map = {
        1: 1,
        2: 1,
        3: 2,
        4: 3,
    }

    total_area_ratio = area_ratios[level]
    num_patches = num_patches_map[level]

    target_area = tw * th
    total_patch_area = int(target_area * total_area_ratio)

    # patch 하나당 면적
    patch_area = max(1, total_patch_area // num_patches)

    # 정사각형에 가깝게 patch 크기 계산
    patch_size = int(math.sqrt(patch_area))
    patch_w = max(1, patch_size)
    patch_h = max(1, patch_size)

    # source 이미지가 너무 작으면 키워줌
    if sw < patch_w or sh < patch_h:
        scale = max(patch_w / sw, patch_h / sh)
        new_sw = int(sw * scale) + 1
        new_sh = int(sh * scale) + 1
        source = source.resize(
            (new_sw, new_sh),
            Image.Resampling.LANCZOS
        )
        sw, sh = source.size

    for _ in range(num_patches):
        # source에서 patch 추출
        src_left = rng.randint(0, max(0, sw - patch_w))
        src_top = rng.randint(0, max(0, sh - patch_h))
        src_right = src_left + patch_w
        src_bottom = src_top + patch_h

        patch = source.crop(
            (src_left, src_top, src_right, src_bottom)
        )

        # target 내 임의 위치에 붙여넣기
        dst_left = rng.randint(0, max(0, tw - patch_w))
        dst_top = rng.randint(0, max(0, th - patch_h))

        target.paste(
            patch,
            (dst_left, dst_top)
        )

    return target