import io
import random
import zlib

from PIL import Image, features

"""
EncodeQuality 변조 (랜덤 품질 인코딩)

이미지를 손실 압축 형식으로 낮은 품질로 한 번 저장했다가 다시 읽어 화질을 떨어뜨린다.
(실제 파일을 쓰지 않고 메모리 안에서만 인코딩/디코딩한다)

압축 형식은 아래 두 가지 중에서 랜덤으로 고른다.
- JPEG: 8x8 블록 단위로 압축한다. 품질이 낮을수록 블록 경계(격자 무늬),
        선 주변의 모기 잡음(mosquito noise), 색 번짐이 생긴다.
        색 정보를 줄이는 방식(4:4:4 / 4:2:2 / 4:2:0)도 랜덤으로 고른다.
- WEBP: 블록 예측 기반으로 압축한다. 품질이 낮을수록 면이 뭉개지고
        경계가 번지며, 그러데이션이 얼룩덜룩해진다.

같은 품질 숫자여도 WEBP가 JPEG보다 덜 손상되므로 형식마다 품질 범위를 따로 둔다.
(형식이 달라도 같은 level이면 비슷한 정도로 손상되도록 맞췄다.)
압축 형식과 색 정보 축소 방식은 이미지마다 랜덤으로 한 번 정해 lv1~lv4에 같이 쓰고,
품질만 level별 범위 안에서 랜덤으로 정한다.
WEBP를 지원하지 않는 Pillow 환경에서는 JPEG만 쓴다.

여러 번 다시 압축하는 JPEGCompression과 달리, 한 번만 저장한 결과이다.
투명도(알파) 채널은 건드리지 않으며, 출력 크기는 원본과 같다.

level별 품질 범위 (JPEG / WEBP, 100이 최고 품질):
level 1: 45~65 / 25~45
level 2: 25~45 / 12~25
level 3: 12~25 / 5~12
level 4: 3~12  / 0~5

같은 이미지와 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# 압축 형식별, level별 품질 범위
QUALITY_RANGES = {
    "JPEG": {
        1: (45, 65),
        2: (25, 45),
        3: (12, 25),
        4: (3, 12),
    },
    "WEBP": {
        1: (25, 45),
        2: (12, 25),
        3: (5, 12),
        4: (0, 5),
    },
}

# JPEG 색 정보 축소 방식 후보 (0: 4:4:4, 1: 4:2:2, 2: 4:2:0)
JPEG_SUBSAMPLING = [0, 1, 2]

# True면 압축 형식과 색 정보 축소 방식을 이미지마다 한 번만 골라 lv1~lv4에 같이 쓴다.
# False면 level마다 따로 고른다.
SAME_STYLE_ALL_LEVELS = True


def encode_decode(image, codec, quality, subsampling):
    # 메모리 안에서 압축 저장한 뒤 다시 읽어 들인다.
    buffer = io.BytesIO()

    if codec == "JPEG":
        image.save(buffer, format="JPEG", quality=quality, subsampling=subsampling)
    else:
        image.save(buffer, format="WEBP", quality=quality, method=4)

    buffer.seek(0)

    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def EncodeQuality(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in QUALITY_RANGES["JPEG"]:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    # 방식을 고르는 style_rng는 이미지 내용만으로 시드를 만들어
    # lv1~lv4가 같은 방식을 쓰고 강도만 달라지게 한다.
    if rng is None:
        image_seed = zlib.crc32(image.tobytes())
        rng = random.Random(image_seed + level)
        style_rng = random.Random(image_seed) if SAME_STYLE_ALL_LEVELS else rng
    else:
        style_rng = rng

    # 압축 형식과 색 정보 축소 방식은 이미지마다 랜덤으로 결정
    codecs = ["JPEG", "WEBP"] if features.check("webp") else ["JPEG"]
    codec = style_rng.choice(codecs)
    subsampling = style_rng.choice(JPEG_SUBSAMPLING)

    # 품질은 level별 범위 안에서 랜덤으로 결정
    quality = rng.randint(*QUALITY_RANGES[codec][level])

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    EncodeQuality = encode_decode(
        image.convert("RGB"),
        codec,
        quality,
        subsampling
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        EncodeQuality.putalpha(alpha)

    return EncodeQuality
