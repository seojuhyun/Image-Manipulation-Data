# AnimeGANv2 pretrained model을 사용하여 입력 이미지를 animation style로 변환하는 코드

from PIL import Image
import torch
import torchvision.transforms as T


# ============================================================
# 모델 cache
# ============================================================

_MODEL_CACHE = {}


def _get_model(style_name: str, device: torch.device):
    """
    AnimeGANv2 pretrained model을 최초 1회만 로드하고
    이후에는 cache된 모델을 재사용한다.
    """

    cache_key = (
        style_name,
        str(device)
    )

    if cache_key not in _MODEL_CACHE:

        model = torch.hub.load(
            "bryandlee/animegan2-pytorch:main",
            "generator",
            pretrained=style_name,
            trust_repo=True
        )

        model = model.to(device)
        model.eval()

        _MODEL_CACHE[cache_key] = model

    return _MODEL_CACHE[cache_key]


# ============================================================
# Animation
# ============================================================

def Animation_AnimeGAN(
    image: Image.Image,
    level: int,
    aux_image: Image.Image | None = None,
    seed: int | None = None
) -> Image.Image:
    """
    AnimeGANv2 기반 Animation 변조

    pretrained AnimeGANv2 generator를 이용해
    입력 이미지를 animation/anime style로 변환한다.

    level은 강도를 의미하지 않고,
    서로 다른 pretrained style variant를 의미한다.

    level 1 -> paprika
    level 2 -> face_paint_512_v1
    level 3 -> face_paint_512_v2
    level 4 -> celeba_distill
    """

    if level not in {1, 2, 3, 4}:
        raise ValueError(
            "level은 1~4 중 하나여야 합니다."
        )

    image = image.convert("RGB")

    # --------------------------------------------------------
    # level별 pretrained AnimeGAN style
    # --------------------------------------------------------

    style_map = {
        1: "paprika",
        2: "face_paint_512_v1",
        3: "face_paint_512_v2",
        4: "celeba_distill",
    }

    style_name = style_map[level]

    # --------------------------------------------------------
    # GPU 사용 가능하면 GPU, 아니면 CPU
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = _get_model(
        style_name,
        device
    )

    # --------------------------------------------------------
    # PIL -> Tensor
    # AnimeGAN 입력 범위: [-1, 1]
    # --------------------------------------------------------

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    tensor = transform(
        image
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # inference
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            tensor
        )

    # --------------------------------------------------------
    # [-1, 1] -> [0, 1]
    # --------------------------------------------------------

    output = (
        output
        .squeeze(0)
        .detach()
        .cpu()
    )

    output = (
        output * 0.5 + 0.5
    )

    output = torch.clamp(
        output,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Tensor -> PIL
    # --------------------------------------------------------

    result = T.ToPILImage()(
        output
    )

    # 원본 크기 유지
    result = result.resize(
        image.size,
        Image.Resampling.LANCZOS
    )

    return result