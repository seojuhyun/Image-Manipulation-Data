import random
import zlib

from PIL import Image

"""
Dislocation 변조 (패치 위치 랜덤 이동)

이미지에서 직사각형 패치를 여러 개 잘라내어, 각각 원래 자리와 다른
랜덤한 위치에 붙여 넣는다.

패치의 크기, 가로/세로 비율, 잘라내는 위치, 붙이는 위치는 모두 랜덤이다.
패치는 항상 원본 이미지에서 잘라내므로(이미 붙인 결과에서 다시 자르지 않음)
패치 안의 내용은 원본 그대로이고, 크기를 바꾸거나 회전시키지 않는다.
패치가 붙지 않은 곳은 원본 그대로 남고, 나중에 붙인 패치가 먼저 붙인
패치를 덮을 수 있다.
출력 크기는 원본과 같다.

level이 높을수록 옮기는 패치 수가 많아져, 원본이 남는 영역이 줄어든다.

level별 패치 개수 범위:
level 1: 2~3개
level 2: 5~8개
level 3: 12~18개
level 4: 25~35개

패치 한 변의 크기는 이미지 가로/세로의 10~40% 사이에서 각각 랜덤으로 정한다.

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 패치 개수 범위
PATCH_COUNTS = {
    1: (2, 3),
    2: (5, 8),
    3: (12, 18),
    4: (25, 35),
}

# 패치 한 변의 크기 범위 (이미지 가로/세로 대비)
PATCH_SIZE_RANGE = (0.10, 0.40)

# 패치가 원래 자리에서 최소한 이만큼은 이동해야 한다 (패치 크기 대비)
MIN_SHIFT = 0.5


def Dislocation(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in PATCH_COUNTS:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # 옮길 패치 개수를 랜덤으로 결정
    num_patches = rng.randint(*PATCH_COUNTS[level])

    Dislocation = image.copy()

    for _ in range(num_patches):

        # 패치 크기를 가로/세로 각각 랜덤으로 결정
        patch_w = max(1, round(rng.uniform(*PATCH_SIZE_RANGE) * width))
        patch_h = max(1, round(rng.uniform(*PATCH_SIZE_RANGE) * height))

        # 잘라낼 위치를 랜덤으로 결정
        src_x = rng.randint(0, width - patch_w)
        src_y = rng.randint(0, height - patch_h)

        # 붙일 위치를 랜덤으로 결정
        # (원래 자리와 충분히 떨어진 곳이 나올 때까지 다시 뽑는다)
        for _ in range(20):
            dst_x = rng.randint(0, width - patch_w)
            dst_y = rng.randint(0, height - patch_h)

            moved_enough = (
                abs(dst_x - src_x) >= MIN_SHIFT * patch_w
                or abs(dst_y - src_y) >= MIN_SHIFT * patch_h
            )

            if moved_enough:
                break

        # 원본에서 패치를 잘라 새 위치에 붙인다.
        patch = image.crop(
            (src_x, src_y, src_x + patch_w, src_y + patch_h)
        )

        Dislocation.paste(patch, (dst_x, dst_y))

    return Dislocation