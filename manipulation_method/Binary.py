import cv2
import numpy as np
from PIL import Image

"""
Binary 변조 (여러 방식의 이진화)

이미지를 흑백(밝기)으로 바꾼 뒤, 임계값을 기준으로 밝기 값을 잘라낸다.
level마다 서로 다른 방식을 쓰며, 뒤로 갈수록 원본의 명암 정보가 더 많이 사라진다.

level 1: TRUNC (윗부분 자르기)
    임계값보다 밝은 픽셀을 모두 임계값으로 낮춘다.
    밝은 영역만 한 가지 회색으로 평평해지고, 어두운 쪽의 명암은 그대로 남는다.

level 2: TOZERO (아랫부분 자르기)
    임계값보다 어두운 픽셀을 모두 검은색(0)으로 만든다.
    어두운 영역만 검게 뭉개지고, 밝은 쪽의 명암은 그대로 남는다.

level 3: BINARY (전역 이진화)
    임계값보다 밝으면 흰색(255), 아니면 검은색(0)으로 만든다.
    이미지 전체가 흑과 백 두 값으로만 표현된다.

level 4: ADAPTIVE (적응형 이진화)
    이미지 전체에 하나의 임계값을 쓰지 않고, 픽셀마다 주변 영역의
    가우시안 가중 평균을 임계값으로 삼아 흑과 백으로 나눈다.
    면은 대부분 흰색이 되고 윤곽선과 질감만 검게 남아 펜 선화처럼 된다.

level 1~3의 임계값은 이미지마다 자동으로 정한다.
Otsu 방법으로 픽셀을 어두운 무리와 밝은 무리로 나눈 뒤,
두 무리의 평균 밝기의 한가운데 값을 임계값으로 쓴다.
(한 가지 밝기뿐인 이미지처럼 무리를 나눌 수 없으면 127을 쓴다.)

랜덤 요소가 없어, 같은 입력과 level에 대해 항상 같은 결과가 나온다.
출력은 RGB이고(투명도가 있는 이미지는 RGBA), 세 채널의 값이 모두 같다.
출력 크기는 원본과 같다.
"""

# level 1~3에서 쓰는 임계값 처리 방식
THRESHOLD_TYPES = {
    1: cv2.THRESH_TRUNC,
    2: cv2.THRESH_TOZERO,
    3: cv2.THRESH_BINARY,
}

# level 4(적응형 이진화)의 설정
ADAPTIVE_BLOCK_SIZE = 11   # 주변 영역의 한 변 크기 (픽셀, 홀수)
ADAPTIVE_C = 2             # 주변 평균에서 빼는 값 (클수록 흰색이 많아진다)


def auto_threshold(gray):
    # 이미지의 밝기 분포에서 임계값을 자동으로 정한다.

    # Otsu 방법으로 어두운 무리와 밝은 무리를 나누는 경계를 구한다.
    otsu, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    dark = gray[gray <= otsu]
    bright = gray[gray > otsu]

    # 무리를 나눌 수 없으면 중간값 127을 쓴다.
    if dark.size == 0 or bright.size == 0:
        return 127

    # 두 무리의 평균 밝기의 한가운데 값
    return int(round((float(dark.mean()) + float(bright.mean())) / 2.0))


def Binary(
    image: Image.Image,
    level: int,
    aux_image: Image.Image = None,
    **kwargs
) -> Image.Image:

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    # 투명도(알파) 채널이 있으면 따로 보관해 둔다.
    has_alpha = "A" in image.getbands()
    alpha = image.getchannel("A") if has_alpha else None

    # 흑백(밝기 0~255)으로 변환
    gray = np.asarray(image.convert("L"))

    if level in THRESHOLD_TYPES:
        # 이미지의 밝기 분포에서 임계값을 자동으로 정한다.
        threshold = auto_threshold(gray)

        # level별 방식으로 임계값 처리
        _, result = cv2.threshold(
            gray, threshold, 255, THRESHOLD_TYPES[level]
        )

    else:
        # 픽셀마다 주변 영역의 가중 평균을 임계값으로 쓰는 적응형 이진화
        result = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=ADAPTIVE_BLOCK_SIZE,
            C=ADAPTIVE_C
        )

    # 1채널 결과를 3채널 RGB로 만든다.
    Binary = Image.fromarray(result).convert("RGB")

    # 보관해 둔 투명도 채널을 다시 붙인다.
    if has_alpha:
        Binary.putalpha(alpha)

    return Binary