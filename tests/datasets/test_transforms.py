"""Test data transforms."""

import pytest
import torch

from yoke.datasets import transforms


def test_resize_pad_crop() -> None:
    """Ensure resize pad crop transform works as intended."""
    image_size = (1120, 400)
    scale_factor = 0.5
    scaled_image_size = (511, 230)  # arbitrary choice to test pad and crop
    tform = transforms.ResizePadCrop(
        interp_kwargs={"scale_factor": scale_factor}, scaled_image_size=scaled_image_size
    )
    image = torch.randn(
        (2, 3, 4, *image_size)
    )  # [batch, sequence length, variables, Y, X]
    out = tform(image)
    assert out.shape[-2:] == scaled_image_size, (
        "ResizePadCrop() output not the expected size!"
    )


def test_resize_pad_crop_top_left_positions() -> None:
    """Exercise the 'top'/'left' pad-position crop branches."""
    image_size = (1120, 400)
    scale_factor = 0.5
    scaled_image_size = (511, 230)
    tform = transforms.ResizePadCrop(
        interp_kwargs={"scale_factor": scale_factor},
        scaled_image_size=scaled_image_size,
        pad_position=("top", "left"),
    )
    image = torch.randn((2, 3, 4, *image_size))
    out = tform(image)
    assert out.shape[-2:] == scaled_image_size, (
        "ResizePadCrop() output not the expected size for top/left positions!"
    )


@pytest.mark.parametrize(
    "pad_position",
    [
        ("bottom", "right"),
        ("bottom", "left"),
        ("top", "right"),
        ("top", "left"),
    ],
)
def test_resize_pad_crop_pixel_correctness(
    pad_position: tuple[str, str],
) -> None:
    """Verify padding is placed and cropped correctly for all pad positions.

    A smaller image is upsized (via padding) to a larger `scaled_image_size` with
    no interpolation scaling. We then assert that the *padding* (zeros) lands on
    the requested edges and the original image content is preserved on the
    opposite edges (i.e. no real content is cropped away).
    """
    # Original content is all ones so padding (zeros) is easy to distinguish.
    orig_h, orig_w = 5, 7
    pad_h, pad_w = 3, 4
    scaled_image_size = (orig_h + pad_h, orig_w + pad_w)

    tform = transforms.ResizePadCrop(
        interp_kwargs={"scale_factor": 1.0},
        scaled_image_size=scaled_image_size,
        pad_mode="constant",
        pad_value=0.0,
        pad_position=pad_position,
    )
    image = torch.ones((1, 1, orig_h, orig_w))
    out = tform(image)

    assert out.shape[-2:] == scaled_image_size, (
        f"Unexpected output size for pad_position={pad_position}!"
    )

    # Determine which rows/cols should be padding (zeros).
    if pad_position[0] == "top":
        pad_rows = slice(0, pad_h)
        content_rows = slice(pad_h, None)
    else:  # "bottom"
        pad_rows = slice(orig_h, None)
        content_rows = slice(0, orig_h)

    if pad_position[1] == "left":
        pad_cols = slice(0, pad_w)
        content_cols = slice(pad_w, None)
    else:  # "right"
        pad_cols = slice(orig_w, None)
        content_cols = slice(0, orig_w)

    # The padded rows/cols must be all zeros.
    assert torch.all(out[..., pad_rows, :] == 0.0), (
        f"Row padding not on the expected edge for pad_position={pad_position}!"
    )
    assert torch.all(out[..., :, pad_cols] == 0.0), (
        f"Column padding not on the expected edge for pad_position={pad_position}!"
    )

    # The original content region must be preserved (all ones), meaning no real
    # image content was cropped away.
    assert torch.all(out[..., content_rows, content_cols] == 1.0), (
        f"Original content not preserved for pad_position={pad_position}!"
    )
