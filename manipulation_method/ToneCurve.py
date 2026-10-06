import random
import zlib

from PIL import Image

"""
ToneCurve 변조 (랜덤 톤 커브)

"입력 밝기 -> 출력 밝기" 대응 곡선(톤 커브)을 랜덤하게 휘어서,
이미지의 어두운 영역과 밝은 영역의 밝기를 따로 바꾼다.

원래의 톤 커브는 입력과 출력이 같은 직선이다.
여기에 조절점 두 개를 두고 위아래로 움직여 곡선을 만든다.
- 어두운 쪽 조절점 (기본 높이 1/3): 올리면 어두운 영역이 밝아지고,
  내리면 더 어두워진다.
- 밝은 쪽 조절점 (기본 높이 2/3): 올리면 밝은 영역이 더 밝아지고,
  내리면 어두워진다.

두 조절점을 움직이는 방향의 조합에 따라 결과가 달라진다.
- 둘 다 올림:                 전체적으로 밝아진다.
- 둘 다 내림:                 전체적으로 어두워진다.
- 어두운 쪽 내림, 밝은 쪽 올림: 대비가 강해진다 (S자 곡선).
- 어두운 쪽 올림, 밝은 쪽 내림: 대비가 약해져 뿌옇게 된다.

곡선은 3차 베지어 곡선으로 부드럽게 이어지며,
검은색(0)과 흰색(255)은 변하지 않고, 밝기의 순서가 뒤집히는 일도 없다.
같은 곡선을 R, G, B 채널에 똑같이 적용한다.

조절점을 움직이는 거리는 level별 범위 안에서, 방향은 랜덤으로 정한다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 조절점 이동 거리 범위 (전체 밝기 범위를 1로 볼 때):
level 1: 0.04~0.08
level 2: 0.08~0.16
level 3: 0.16~0.24
level 4: 0.24~0.33

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 조절점 이동 거리 범위 (전체 밝기 범위를 1로 볼 때)
# 1/3을 넘기면 조절점이 0~1 범위 끝에 닿아 더 움직이지 않는다.
SHIFT_RANGES = {
    1: (0.04, 0.08),
    2: (0.08, 0.16),
    3: (0.16, 0.24),
    4: (0.24, 0.33),
}

# 조절점의 기본 높이 (이 값일 때 톤 커브는 입력 = 출력인 직선이 된다)
BASE_LOW = 1 / 3
BASE_HIGH = 2 / 3


def ToneCurve(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in SHIFT_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # 조절점을 움직일 거리(level별 범위)와 방향(부호)을 랜덤으로 결정
    def random_shift():
        value = rng.uniform(*SHIFT_RANGES[level])
        return value if rng.random() < 0.5 else -value

    # 어두운 쪽 / 밝은 쪽 조절점의 높이 (0~1 범위를 벗어나지 않게 한다)
    low = min(max(BASE_LOW + random_shift(), 0.0), 1.0)
    high = min(max(BASE_HIGH + random_shift(), 0.0), 1.0)

    # 어두운 쪽 조절점이 밝은 쪽보다 높아지면 밝기 순서가 뒤집히므로,
    # 그런 경우에는 두 조절점을 가운데 높이로 맞춘다.
    if low > high:
        low = high = (low + high) / 2.0

    # 0~255의 각 값이 변환 후 어떤 값이 되는지 미리 계산해 둔 표
    # (조절점 높이가 0, low, high, 1인 3차 베지어 곡선)
    table = []

    for value in range(256):
        t = value / 255.0
        curve = (
            3 * (1 - t) ** 2 * t * low
            + 3 * (1 - t) * t ** 2 * high
            + t ** 3
        )
        table.append(min(255, max(0, round(curve * 255.0))))

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    # 흑백(L)은 그대로, 그 외에는 RGB로 바꿔서 처리한다.
    base = image if image.mode == "L" else image.convert("RGB")

    # 표를 모든 채널에 적용한다.
    ToneCurve = base.point(table * len(base.getbands()))

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        ToneCurve.putalpha(alpha)

    return ToneCurve