import random
import zlib

import numpy as np
from PIL import Image

"""
DropChannel 변조 (랜덤 채널 제거)

이미지의 R, G, B 채널 중 일부를 랜덤으로 골라 값을 모두 0으로 만든다.

- 1개를 제거하면 남은 두 채널의 색만 섞여 보인다.
  (R 제거 -> 청록빛, G 제거 -> 자줏빛, B 제거 -> 노란빛)
- 2개를 제거하면 한 채널만 남아, 이미지 전체가 빨강/초록/파랑 중
  한 가지 색의 명암으로만 표현된다.

형태와 위치는 그대로이고 색 정보만 사라진다.
제거할 채널의 개수는 level로 정해지고, 어떤 채널을 제거할지는 랜덤이다.

출력은 RGB이고(투명도가 있는 이미지는 RGBA), 출력 크기는 원본과 같다.

level별 제거하는 채널 수:
level 1: 1개
level 2: 1개
level 3: 1개 또는 2개 (랜덤)
level 4: 2개 (한 채널만 남김)

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 제거할 채널 수의 후보 (후보가 여러 개면 그중에서 랜덤으로 고른다)
DROP_COUNTS = {
    1: [1],
    2: [1],
    3: [1, 2],
    4: [2],
}


def DropChannel(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in DROP_COUNTS:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    # (높이, 너비, 3) 배열. 마지막 축이 R, G, B 채널이다.
    channels = np.array(image.convert("RGB"))

    # 제거할 채널의 개수와 종류를 랜덤으로 결정
    drop_count = rng.choice(DROP_COUNTS[level])
    drop_indices = rng.sample([0, 1, 2], drop_count)

    # 선택된 채널의 값을 모두 0으로 만든다.
    for idx in drop_indices:
        channels[..., idx] = 0

    DropChannel = Image.fromarray(channels)

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        DropChannel.putalpha(alpha)

    return DropChannel