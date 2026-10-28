import random
import zlib

import numpy as np
from PIL import Image

"""
FDA 변조 (랜덤 Fourier Domain Adaptation)  ※ aux image 필요 (requires_aux=True)

다른 이미지(aux image)의 "분위기"를 푸리에 주파수 공간에서 옮겨 온다.
(Yang & Soatto, "FDA: Fourier Domain Adaptation", CVPR 2020)

1. 원본과 aux image(원본 크기로 맞춤)를 R, G, B 채널마다 푸리에 변환한다.
2. 푸리에 변환 결과를 진폭(각 주파수 성분의 세기)과 위상(성분의 위치 정보)으로 나눈다.
3. 원본 진폭의 가운데 정사각형 영역(= 저주파 성분)을 aux image의 진폭과 섞는다.
       섞인 진폭 = (1 - mix) x 원본 진폭 + mix x aux 진폭
4. 섞인 진폭 + 원본의 위상으로 다시 이미지를 만든다.

위상은 원본 그대로이므로 선, 글자, 형태는 유지되고,
저주파 진폭이 바뀌면서 전체 색감, 밝기, 큰 범위의 명암 분포가 aux image를 닮아 간다.
바꾸는 영역(beta)이 넓어질수록 중간 주파수까지 바뀌어
번진 듯한 얼룩, 물결 모양 잔상, 색 번짐이 생긴다.

원 논문은 진폭을 완전히 바꿔 넣는다(mix = 100%).
이 경우 가장 낮은 주파수(평균 밝기와 색)가 통째로 바뀌어 약한 단계에서도
변화가 너무 커지므로, 섞는 비율(mix)도 level에 따라 키운다.
(진폭을 섞는 방식은 Xu et al., "A Fourier-based Framework for
Domain Generalization", CVPR 2021의 Amplitude Mix와 같다.)
level 4는 원 논문과 같이 거의 완전히 바꿔 넣는다.

바꾸는 정사각형의 크기 = 이미지 짧은 변 x beta (가운데로부터 반 변 길이)
beta와 mix는 level별 범위 안에서 랜덤으로 정한다.
aux image는 runner1(lv1~lv4 같은 aux)로 쓰면 같은 분위기로 강도만 달라진다.

출력은 RGB이고(투명도가 있는 이미지는 RGBA), 출력 크기는 원본과 같다.

level별 beta / mix 범위 (짧은 변 720px 기준 바꾸는 저주파 범위):
level 1: 0.004~0.008 / 25~40%  (가운데에서 ±2~5번째 주파수까지)
level 2: 0.008~0.015 / 40~60%  (±5~10번째)
level 3: 0.015~0.025 / 60~80%  (±10~18번째)
level 4: 0.025~0.040 / 85~100% (±18~28번째)

같은 이미지, 같은 aux image, 같은 level에 대해서는 항상 같은 결과가 나온다.
"""

# level별 FDA 설정
# beta: 바꿀 저주파 정사각형의 반 변 길이 (이미지 짧은 변 대비)
# mix:  저주파 진폭을 aux image 쪽으로 섞는 비율 (1이면 완전히 바꿔 넣음)
CONFIGS = {
    1: {"beta": (0.004, 0.008), "mix": (0.25, 0.40)},
    2: {"beta": (0.008, 0.015), "mix": (0.40, 0.60)},
    3: {"beta": (0.015, 0.025), "mix": (0.60, 0.80)},
    4: {"beta": (0.025, 0.040), "mix": (0.85, 1.00)},
}


def FDA(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    rng: random.Random = None,
    **kwargs
) -> Image.Image:

    if level not in CONFIGS:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    if aux_image is None:
        raise ValueError(
            "FDA는 aux_image가 필요합니다. "
            "(Registry에 requires_aux=True로 등록하세요)"
        )

    width, height = image.size

    # 너무 작은 이미지는 그대로 돌려준다.
    if width < 2 or height < 2:
        return image.copy()

    # rng를 넘겨받지 않으면 이미지 내용 + level로 시드를 만든다.
    if rng is None:
        seed = zlib.crc32(image.tobytes()) + level
        rng = random.Random(seed)

    # level별 범위 안에서 beta와 섞는 비율을 랜덤으로 결정
    config = CONFIGS[level]
    beta = rng.uniform(*config["beta"])
    mix = rng.uniform(*config["mix"])
    border = max(1, int(np.floor(min(width, height) * beta)))

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    source = np.asarray(image.convert("RGB")).astype(np.float64)

    # aux image를 원본과 같은 크기로 맞춘다. (저주파만 쓰므로 비율이 바뀌어도 괜찮다)
    reference = np.asarray(
        aux_image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    ).astype(np.float64)

    # 진폭 스펙트럼의 가운데(저주파) 정사각형 범위
    center_y, center_x = height // 2, width // 2
    y1, y2 = max(0, center_y - border), min(height, center_y + border + 1)
    x1, x2 = max(0, center_x - border), min(width, center_x + border + 1)

    adapted = np.empty_like(source)

    # 메모리를 아끼기 위해 채널마다 따로 처리한다.
    for idx in range(3):

        source_fft = np.fft.fft2(source[..., idx])
        reference_fft = np.fft.fft2(reference[..., idx])

        # 진폭과 위상으로 나눈다.
        amplitude = np.fft.fftshift(np.abs(source_fft))
        phase = np.angle(source_fft)
        reference_amplitude = np.fft.fftshift(np.abs(reference_fft))

        # 저주파 진폭만 aux image의 진폭 쪽으로 섞는다.
        amplitude[y1:y2, x1:x2] = (
            (1.0 - mix) * amplitude[y1:y2, x1:x2]
            + mix * reference_amplitude[y1:y2, x1:x2]
        )
        amplitude = np.fft.ifftshift(amplitude)

        # 섞인 진폭 + 원본 위상으로 다시 이미지를 만든다.
        adapted[..., idx] = np.real(
            np.fft.ifft2(amplitude * np.exp(1j * phase))
        )

    FDA = Image.fromarray(
        np.clip(np.rint(adapted), 0, 255).astype(np.uint8)
    )

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        FDA.putalpha(alpha)

    return FDA
