# 색조 변화가 아니라 watercolor, sketch, brush, abstract 표현 방식으로 이미지를 stylize하는 코드
'''
mkdir -p external
cd external
git clone https://github.com/naoto0804/pytorch-AdaIN.git

cd pytorch-AdaIN
wget https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth
wget https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth


**AdaIN(Adaptive Instance Normalization)**은 한 이미지의 내용(Content)은 최대한 유지하면서, 다른 이미지의 스타일(Style)을 적용하는 딥러닝 기반 Style Transfer 알고리즘
Stylize 기법은 AdaIN(Adaptive Instance Normalization)을 이용하여 무작위로 선택된 참조 이미지의 스타일을 원본 이미지에 적용한다. 
스타일 변환 강도를 조절하는 α는 level 1부터 4까지 각각 0.3, 0.5, 0.7, 0.9로 설정하였다. 
이를 통해 원본 이미지의 콘텐츠 특징을 일부 유지하면서 참조 이미지의 스타일 반영 강도를 단계적으로 증가시킨다.
'''

# random aux image의 style을 현재 이미지에 전달하는 AdaIN 기반 Stylize 코드

from PIL import Image
from manipulation_method.style_transfer_adain import *



def Stylize(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:
    """
    Stylize 변조

    현재 image의 content 구조를 유지하면서
    random aux_image의 style을 전달한다.

    level은 style transfer 강도를 의미한다.

    level 1 -> 40%
    level 2 -> 60%
    level 3 -> 80%
    level 4 -> 100%
    """

    if level not in {1, 2, 3, 4}:
        raise ValueError(
            "level은 1~4 중 하나여야 합니다."
        )

    if aux_image is None:
        raise ValueError(
            "Stylize는 aux_image가 필요합니다."
        )

    alpha = {
        1: 0.30,
        2: 0.50,
        3: 0.70,
        4: 0.90,
    }[level]

    return adain_style_transfer(
        content_image=image,
        style_image=aux_image,
        alpha=alpha
    )