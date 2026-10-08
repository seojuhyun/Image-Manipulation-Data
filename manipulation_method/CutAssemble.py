# 기본 N×N grid를 만든 뒤, 인접한 cell들을 직사각형 block으로 랜덤 병합하고 block 위치를 섞어 재조립하는 코드

from PIL import Image
import random


def _make_random_blocks(
    grid_size: int,
    max_block_h: int,
    max_block_w: int,
    rng: random.Random
):
    """
    grid_size x grid_size 기본 grid를
    겹치지 않는 직사각형 block들로 분할한다.

    각 block은
    (row_start, row_end, col_start, col_end)
    형식으로 저장한다.
    end는 exclusive.
    """

    used = [
        [False] * grid_size
        for _ in range(grid_size)
    ]

    blocks = []

    for r in range(grid_size):
        for c in range(grid_size):

            if used[r][c]:
                continue

            # 현재 위치에서 가능한 최대 높이
            possible_h = 1
            for h in range(1, max_block_h + 1):
                if r + h > grid_size:
                    break

                ok = True
                for rr in range(r, r + h):
                    if used[rr][c]:
                        ok = False
                        break

                if ok:
                    possible_h = h
                else:
                    break

            block_h = rng.randint(1, possible_h)

            # 현재 위치와 선택한 높이에서 가능한 최대 너비
            possible_w = 1
            for w in range(1, max_block_w + 1):
                if c + w > grid_size:
                    break

                ok = True
                for rr in range(r, r + block_h):
                    for cc in range(c, c + w):
                        if used[rr][cc]:
                            ok = False
                            break
                    if not ok:
                        break

                if ok:
                    possible_w = w
                else:
                    break

            block_w = rng.randint(1, possible_w)

            # 가끔 세로/가로가 너무 큰 block를 줄여 다양성 확보
            if block_h > 1 and rng.random() < 0.35:
                block_h -= 1

            if block_w > 1 and rng.random() < 0.35:
                block_w -= 1

            # used 표시
            for rr in range(r, r + block_h):
                for cc in range(c, c + block_w):
                    used[rr][cc] = True

            blocks.append(
                (r, r + block_h, c, c + block_w)
            )

    return blocks


def CutAssemble(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:
    """
    CutAssemble 변조

    이미지를 기본 N×N grid로 나눈 뒤,
    여러 cell을 직사각형 block으로 랜덤 병합하고,
    그 block들의 위치를 섞어 재조립한다.

    예:
    - 어떤 block은 1×1
    - 어떤 block은 2×1
    - 어떤 block은 1×2
    - 어떤 block은 2×2

    level 기준:
    - level 1 -> 4×4 grid, 작은 block 위주
    - level 2 -> 4×4 grid, 2칸 block 증가
    - level 3 -> 5×5 grid, 더 다양한 block
    - level 4 -> 6×6 grid, 가장 다양한 block
    """

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    rng = random.Random(seed)
    image = image.convert("RGB")

    width, height = image.size

    # --------------------------------------------------------
    # 기본 grid 크기
    # --------------------------------------------------------
    grid_size = {
        1: 4,
        2: 4,
        3: 5,
        4: 6,
    }[level]

    # --------------------------------------------------------
    # 최대 병합 크기
    # level이 높을수록 더 큰 block 허용
    # --------------------------------------------------------
    max_block_h = {
        1: 2,
        2: 2,
        3: 3,
        4: 3,
    }[level]

    max_block_w = {
        1: 2,
        2: 3,
        3: 3,
        4: 4,
    }[level]

    # --------------------------------------------------------
    # block layout 생성
    # --------------------------------------------------------
    blocks = _make_random_blocks(
        grid_size=grid_size,
        max_block_h=max_block_h,
        max_block_w=max_block_w,
        rng=rng
    )

    # --------------------------------------------------------
    # block patch 추출
    # --------------------------------------------------------
    patches = []

    for r0, r1, c0, c1 in blocks:

        left = int(c0 * width / grid_size)
        right = int(c1 * width / grid_size)
        top = int(r0 * height / grid_size)
        bottom = int(r1 * height / grid_size)

        patch = image.crop(
            (left, top, right, bottom)
        )

        patches.append(patch)

    # --------------------------------------------------------
    # patch 순서를 섞어서 block 위치 shuffle
    # --------------------------------------------------------
    shuffled_patches = patches[:]
    rng.shuffle(shuffled_patches)

    # --------------------------------------------------------
    # 재조립
    # --------------------------------------------------------
    result = Image.new("RGB", (width, height))

    for (r0, r1, c0, c1), patch in zip(blocks, shuffled_patches):

        left = int(c0 * width / grid_size)
        right = int(c1 * width / grid_size)
        top = int(r0 * height / grid_size)
        bottom = int(r1 * height / grid_size)

        target_w = right - left
        target_h = bottom - top

        patch = patch.resize(
            (target_w, target_h),
            Image.Resampling.LANCZOS
        )

        result.paste(
            patch,
            (left, top)
        )

    return result