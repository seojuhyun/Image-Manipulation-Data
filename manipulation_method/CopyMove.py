# "requires_aux": False

import math
import random
from PIL import Image


def CopyMove(
    image: Image.Image,
    level: int,
    aux_image=None,
    seed: int | None = None
) -> Image.Image:
    """
    CopyMove 변조

    동일 이미지 내부의 영역을 복사하여
    다른 위치에 붙여넣는다.

    level은 전체 이미지 대비 총 변조 면적 비율로 정의한다.

    level 1: 약 5%
    level 2: 약 10%
    level 3: 약 20%
    level 4: 약 30%
    """

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    rng = random.Random(seed)

    result = image.convert("RGB").copy()

    width, height = result.size

    # level에 따라 전체 이미지 대비 총 변조 면적 비율을 5%, 10%, 20%, 30%로 설정
    area_ratios = {
        1: 0.05,
        2: 0.10,
        3: 0.20,
        4: 0.30,
    }

    num_patches_map = {
        1: 1,
        2: 1,
        3: 2,
        4: 3,
    }

    total_area_ratio = area_ratios[level]
    num_patches = num_patches_map[level]

    image_area = width * height
    total_patch_area = int(
        image_area * total_area_ratio
    )

    patch_area = max(
        1,
        total_patch_area // num_patches
    )

    patch_size = int(
        math.sqrt(patch_area)
    )

    patch_w = max(
        1,
        patch_size
    )

    patch_h = max(
        1,
        patch_size
    )

    for _ in range(num_patches):

        # ----------------------------------------------------
        # 복사할 source 영역 선택
        # ----------------------------------------------------

        src_left = rng.randint(
            0,
            max(0, width - patch_w)
        )

        src_top = rng.randint(
            0,
            max(0, height - patch_h)
        )

        src_right = (
            src_left + patch_w
        )

        src_bottom = (
            src_top + patch_h
        )

        patch = result.crop(
            (
                src_left,
                src_top,
                src_right,
                src_bottom
            )
        )


        # ----------------------------------------------------
        # source와 겹치지 않는 destination 위치 선택
        # ----------------------------------------------------

        for _ in range(50):

            dst_left = rng.randint(
                0,
                max(0, width - patch_w)
            )

            dst_top = rng.randint(
                0,
                max(0, height - patch_h)
            )

            overlap_x = (
                dst_left < src_right
                and dst_left + patch_w > src_left
            )

            overlap_y = (
                dst_top < src_bottom
                and dst_top + patch_h > src_top
            )

            if not (
                overlap_x
                and overlap_y
            ):
                break


        # ----------------------------------------------------
        # 복사한 영역 붙여넣기
        # ----------------------------------------------------

        result.paste(
            patch,
            (
                dst_left,
                dst_top
            )
        )

    return result