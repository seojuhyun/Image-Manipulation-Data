# 이미지의 색상과 세부 영역을 크게 뭉쳐 oil painting처럼 표현하는 코드
'''
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'xphoto'))"
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-contrib-python


'''


from PIL import Image
import numpy as np
import cv2


def OilPaint(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    arr = np.array(
        image.convert("RGB")
    )

    size = { # size가 클수록 oilPainting() 효과가 강해짐/ 유화 효과의 주변 픽셀 참조 반경
        1: 3,
        2: 4,
        3: 5,
        4: 6,
    }[level]

    # 먼저 약간 downsample → blocky effect
    h, w = arr.shape[:2]

    scale = { # scale이 작을수록 더 많이 축소했다가 확대하므로 blocky effect가 강해짐/ 원본 해상도 유지 비율
        1: 0.95,
        2: 0.92,
        3: 0.89,
        4: 0.86,
    }[level]

    small = cv2.resize(
        arr,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    result = cv2.xphoto.oilPainting(
        small,
        size,
        1
    )

    result = cv2.resize(
        result,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    return Image.fromarray(result)
'''
# 비교적 약한 oil painting 효과를 적용하여 level이 올라갈수록 조금씩 더 painterly하게 만드는 코드

from PIL import Image
import numpy as np
import cv2


def OilPaint(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:
    """
    OilPaint 변조

    예시 이미지처럼 완전히 뭉개지지 않고,
    색상 영역이 조금씩 유화처럼 정리되는 방향으로 구성.

    level 1 -> 약한 유화 느낌 (기존 코드의 강한 lv보다 훨씬 약함)
    level 4 -> 가장 강하지만 여전히 과도하지 않음
    """

    if level not in {1, 2, 3, 4}:
        raise ValueError("level은 1~4 중 하나여야 합니다.")

    arr = np.array(image.convert("RGB"))

    # -----------------------------
    # level이 올라갈수록 조금 더 smoothing
    # -----------------------------
    sigma = {
        1: 40,
        2: 55,
        3: 75,
        4: 95,
    }[level]

    smooth = cv2.bilateralFilter(
        arr,
        d=7,
        sigmaColor=sigma,
        sigmaSpace=sigma
    )

    # -----------------------------
    # color quantization
    # 너무 세지 않게 조정
    # 숫자가 커질수록 색상 단순화가 강해짐
    # -----------------------------
    div = {
        1: 18,
        2: 22,
        3: 26,
        4: 30,
    }[level]

    result = (smooth // div) * div

    # -----------------------------
    # 살짝만 painterly하게 보이도록 median blur
    # -----------------------------
    blur_k = {
        1: 3,
        2: 3,
        3: 5,
        4: 5,
    }[level]

    result = cv2.medianBlur(result, blur_k)

    return Image.fromarray(result)
'''