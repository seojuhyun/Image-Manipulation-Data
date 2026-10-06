import random
import zlib

import numpy as np
from PIL import Image

"""
ChangeChan 변조 (랜덤 채널 이동 / 교환 / 반전)

이미지의 R, G, B 채널에 아래 세 가지 연산 중 일부를 랜덤으로 골라 적용한다.

- shift (이동):  채널마다 서로 다른 방향과 거리로 위치를 옮긴다.
                 채널끼리 어긋나면서 윤곽선이 여러 색으로 번져 보인다.
                 이미지 밖으로 밀려난 부분은 반대쪽 가장자리로 넘어온다.
- swap (교환):   세 채널의 순서를 바꾼다. (예: R, G, B -> B, R, G)
                 형태는 그대로이고 색만 바뀐다.
- invert (반전): 일부 채널의 밝기를 뒤집는다. (값 v를 255 - v로)
                 세 채널을 모두 뒤집으면 필름 네거티브처럼 된다.

level이 높을수록 함께 적용하는 연산의 개수가 늘고,
이동 거리와 반전하는 채널 수도 커진다.
어떤 연산을 고를지, 채널별 이동 방향과 거리, 바뀐 채널 순서,
반전할 채널은 모두 랜덤이다.
여러 연산을 함께 적용할 때는 반전 -> 이동 -> 교환 순서로 한다.

출력은 RGB이고(투명도가 있는 이미지는 RGBA), 출력 크기는 원본과 같다.

level별 연산 개수 / 최대 이동 거리(이미지 짧은 변 대비) / 반전 채널 수:
level 1: 1개 / 2%  / 1개
level 2: 1개 / 4%  / 1~2개
level 3: 2개 / 7%  / 2개
level 4: 3개 / 10% / 2~3개

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 설정
# num_ops:      세 연산(shift, swap, invert) 중 몇 개를 적용할지
# max_shift:    채널 이동 거리의 최댓값 (이미지 짧은 변 대비)
# invert_count: 반전할 채널 수의 범위
CONFIGS = {
    1: {"num_ops": 1, "max_shift": 0.02, "invert_count": (1, 1)},
    2: {"num_ops": 1, "max_shift": 0.04, "invert_count": (1, 2)},
    3: {"num_ops": 2, "max_shift": 0.07, "invert_count": (2, 2)},
    4: {"num_ops": 3, "max_shift": 0.10, "invert_count": (2, 3)},
}

# 채널 순서를 바꾸는 방법 (원래 순서 (0, 1, 2)는 제외)
CHANNEL_ORDERS = [
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
]


def ChangeChan(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in CONFIGS:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    width, height = image.size

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    config = CONFIGS[level]

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    # (높이, 너비, 3) 배열. 마지막 축이 R, G, B 채널이다.
    channels = np.array(image.convert("RGB"))

    # 세 연산 중 이번에 적용할 것을 랜덤으로 고른다.
    ops = set(rng.sample(["shift", "swap", "invert"], config["num_ops"]))

    # 1. invert: 랜덤으로 고른 채널의 밝기를 뒤집는다.
    if "invert" in ops:
        invert_count = rng.randint(*config["invert_count"])

        for idx in rng.sample(range(3), invert_count):
            channels[..., idx] = 255 - channels[..., idx]

    # 2. shift: 채널마다 서로 다른 방향과 거리로 옮긴다.
    if "shift" in ops:
        max_shift = max(1.0, config["max_shift"] * min(width, height))

        # 최댓값의 50~100% 거리로, 방향(부호)은 랜덤으로 정한다.
        def shift_amount():
            value = max(1, round(rng.uniform(0.5, 1.0) * max_shift))
            return value if rng.random() < 0.5 else -value

        for idx in range(3):
            dx = shift_amount()
            dy = shift_amount()

            # 밀려난 부분은 반대쪽 가장자리로 넘어온다.
            channels[..., idx] = np.roll(
                channels[..., idx],
                shift=(dy, dx),
                axis=(0, 1)
            )

    # 3. swap: 채널 순서를 바꾼다.
    if "swap" in ops:
        order = rng.choice(CHANNEL_ORDERS)
        channels = channels[..., list(order)]

    ChangeChan = Image.fromarray(np.ascontiguousarray(channels))

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        ChangeChan.putalpha(alpha)

    return ChangeChan