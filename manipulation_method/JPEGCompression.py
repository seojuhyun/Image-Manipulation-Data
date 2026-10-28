import io
import random
import zlib

from PIL import Image

"""
JPEGCompression 변조 (JPEG 재압축)

이미지를 JPEG로 압축했다가 다른 품질로 다시 압축하기를 여러 번 반복한다.
(실제 파일을 쓰지 않고 메모리 안에서만 인코딩/디코딩한다)

한 번 압축할 때마다 8x8 블록 단위로 정보가 버려지고, 다른 품질로 다시 압축하면
이전 압축의 손상 위에 새 손상이 쌓인다(이중 양자화).
블록 경계(격자 무늬), 선 주변의 모기 잡음, 색 번짐이 누적되어 나타난다.
SNS에 올렸다 내려받기를 반복한 이미지나, 조작 흔적을 감추려고(anti-forensics)
다시 압축한 이미지와 같은 형태이다.

압축 횟수(2~3회)는 랜덤이고, 회차마다 품질을 level별 범위 안에서 따로 뽑는다.
색 정보를 줄이는 방식(4:4:4 / 4:2:2 / 4:2:0)도 회차마다 랜덤으로 고른다.

한 번만 저장하는 EncodeQuality와 달리 "여러 번 다시 저장된 흔적"이 남는다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 회차당 JPEG 품질 범위 (100이 최고 품질):
level 1: 70~90
level 2: 50~70
level 3: 30~50
level 4: 10~30

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 회차당 JPEG 품질 범위
QUALITY_RANGES = {
    1: (70, 90),
    2: (50, 70),
    3: (30, 50),
    4: (10, 30),
}

# 압축 횟수 범위
ROUND_RANGE = (2, 3)

# 색 정보 축소 방식 후보 (0: 4:4:4, 1: 4:2:2, 2: 4:2:0)
SUBSAMPLING_CHOICES = [0, 1, 2]


def JPEGCompression(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in QUALITY_RANGES:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    compressed = image.convert("RGB")

    # 압축 횟수를 정하고, 회차마다 품질과 색 정보 축소 방식을 랜덤으로 골라 다시 압축한다.
    rounds = rng.randint(*ROUND_RANGE)

    for _ in range(rounds):
        quality = rng.randint(*QUALITY_RANGES[level])
        subsampling = rng.choice(SUBSAMPLING_CHOICES)

        buffer = io.BytesIO()
        compressed.save(buffer, format="JPEG", quality=quality, subsampling=subsampling)
        buffer.seek(0)

        with Image.open(buffer) as decoded:
            compressed = decoded.convert("RGB")

    JPEGCompression = compressed

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        JPEGCompression.putalpha(alpha)

    return JPEGCompression
