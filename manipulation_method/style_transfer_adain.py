# pretrained AdaIN 모델을 이용해 content 이미지에 reference 이미지의 style을 적용하는 공통 코드

from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms


# ------------------------------------------------------------
# project root
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ADAIN_ROOT = (
    PROJECT_ROOT
    / "external"
    / "pytorch-AdaIN"
)

VGG_PATH = ADAIN_ROOT / "vgg_normalised.pth"
DECODER_PATH = ADAIN_ROOT / "decoder.pth"


# AdaIN repository의 net.py / function.py import
if str(ADAIN_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ADAIN_ROOT)
    )

import net
from function import adaptive_instance_normalization


# ------------------------------------------------------------
# global model cache
# ------------------------------------------------------------

_VGG = None
_DECODER = None
_DEVICE = None


def _get_device():
    """
    CUDA 사용 가능하면 GPU,
    그렇지 않으면 CPU를 사용한다.
    """

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def _load_models():
    """
    AdaIN encoder / decoder를 최초 한 번만 로드한다.
    """

    global _VGG
    global _DECODER
    global _DEVICE

    if _VGG is not None and _DECODER is not None:
        return _VGG, _DECODER, _DEVICE

    if not VGG_PATH.exists():
        raise FileNotFoundError(
            f"VGG weight를 찾을 수 없습니다: {VGG_PATH}"
        )

    if not DECODER_PATH.exists():
        raise FileNotFoundError(
            f"Decoder weight를 찾을 수 없습니다: {DECODER_PATH}"
        )

    _DEVICE = _get_device()

    # pretrained VGG
    vgg = net.vgg

    vgg.load_state_dict(
        torch.load(
            VGG_PATH,
            map_location="cpu"
        )
    )

    # relu4_1까지 사용
    vgg = torch.nn.Sequential(
        *list(vgg.children())[:31]
    )

    # pretrained decoder
    decoder = net.decoder

    decoder.load_state_dict(
        torch.load(
            DECODER_PATH,
            map_location="cpu"
        )
    )

    vgg = vgg.to(_DEVICE)
    decoder = decoder.to(_DEVICE)

    vgg.eval()
    decoder.eval()

    for param in vgg.parameters():
        param.requires_grad = False

    for param in decoder.parameters():
        param.requires_grad = False

    _VGG = vgg
    _DECODER = decoder

    return _VGG, _DECODER, _DEVICE


def _pil_to_tensor(
    image: Image.Image,
    max_size: int = 768
):
    """
    PIL 이미지를 AdaIN 입력 Tensor로 변환한다.

    너무 큰 이미지는 추론 속도를 위해
    max_size 이하로 임시 resize한다.
    """

    image = image.convert("RGB")

    w, h = image.size

    scale = min(
        1.0,
        max_size / max(w, h)
    )

    new_w = max(
        1,
        int(w * scale)
    )

    new_h = max(
        1,
        int(h * scale)
    )

    image = image.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )

    transform = transforms.ToTensor()

    tensor = transform(
        image
    ).unsqueeze(0)

    return tensor


def adain_style_transfer(
    content_image: Image.Image,
    style_image: Image.Image,
    alpha: float = 1.0,
    max_size: int = 768
) -> Image.Image:
    """
    content 이미지의 구조를 유지하면서
    style 이미지의 feature statistics를 전달한다.

    alpha:
        0.0 -> content 유지
        1.0 -> style 최대 적용
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha는 0.0~1.0 사이여야 합니다."
        )

    original_size = content_image.size

    vgg, decoder, device = _load_models()

    content = _pil_to_tensor(
        content_image,
        max_size=max_size
    ).to(device)

    style = _pil_to_tensor(
        style_image,
        max_size=max_size
    ).to(device)

    with torch.no_grad():

        # content / style feature 추출
        content_feature = vgg(content)
        style_feature = vgg(style)

        # style statistics를 content에 적용
        transformed_feature = adaptive_instance_normalization(
            content_feature,
            style_feature
        )

        # alpha로 원본 content feature와 style feature 조절
        transformed_feature = (
            alpha * transformed_feature
            +
            (1.0 - alpha) * content_feature
        )

        output = decoder(
            transformed_feature
        )

    output = (
        output
        .squeeze(0)
        .detach()
        .cpu()
        .clamp(0, 1)
    )

    result = transforms.ToPILImage()(
        output
    )

    # 최종 출력은 원본 크기로 복원
    result = result.resize(
        original_size,
        Image.Resampling.LANCZOS
    )

    return result