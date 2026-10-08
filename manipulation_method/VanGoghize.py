# 고정된 Vincent van Gogh 작품을 style reference로 사용해 Van Gogh 화풍으로 변환하는 코드

'''
mkdir -p external
cd external
git clone https://github.com/naoto0804/pytorch-AdaIN.git

cd pytorch-AdaIN
wget https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth
wget https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth


**AdaIN(Adaptive Instance Normalization)**은 한 이미지의 내용(Content)은 최대한 유지하면서, 다른 이미지의 스타일(Style)을 적용하는 딥러닝 기반 Style Transfer 알고리즘
'''
from pathlib import Path

from PIL import Image
from manipulation_method.style_transfer_adain import *


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VAN_GOGH_STYLE_PATH = (
    PROJECT_ROOT
    / "style_reference"
    / "van_gogh.jpg"
)


_VAN_GOGH_STYLE = None


def _get_van_gogh_style():
    """
    Van Gogh reference 이미지를 최초 한 번만 로드한다.
    """

    global _VAN_GOGH_STYLE

    if _VAN_GOGH_STYLE is None:

        if not VAN_GOGH_STYLE_PATH.exists():
            raise FileNotFoundError(
                f"Van Gogh style 이미지를 찾을 수 없습니다: "
                f"{VAN_GOGH_STYLE_PATH}"
            )

        with Image.open(
            VAN_GOGH_STYLE_PATH
        ) as img:

            _VAN_GOGH_STYLE = (
                img.convert("RGB").copy()
            )

    return _VAN_GOGH_STYLE


def VanGoghize(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:
    """
    VanGoghize 변조

    고정된 Vincent van Gogh painting을
    style reference로 사용한다.

    level은 Van Gogh style 적용 강도를 의미한다.

    level 1 -> 40%
    level 2 -> 60%
    level 3 -> 80%
    level 4 -> 100%
    """

    if level not in {1, 2, 3, 4}:
        raise ValueError(
            "level은 1~4 중 하나여야 합니다."
        )

    alpha = {
        1: 0.30,
        2: 0.50,
        3: 0.70,
        4: 0.90,
    }[level]

    style_image = _get_van_gogh_style()

    return adain_style_transfer(
        content_image=image,
        style_image=style_image,
        alpha=alpha
    )