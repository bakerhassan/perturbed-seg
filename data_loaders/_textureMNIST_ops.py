from PIL import Image
from shared import utils
import torch
from torchvision.transforms.functional import pil_to_tensor

def extract_patch(im, patchsize, rng, xy=None):
    """ Extract a patch from the texture image.

    Args
    ----
    im: torch.Tensor
        Texture image
    patchsize: int
        Size of the patch to be extracted from `im`.
    rng: numpy.random.Generator
        The random generator to be used when extracting the patch
    xy: numpy.ndarray
        An nx2 array of (x,y) coordinates of the upper left corner of the
        n patches that we want to extract. If `xy` is given, `rng` is ignored.

    Returns
    -------
    patch: torch.Tensor
        The random patch extracted from `im`.
    """
    rng = utils.check_rng(rng)
    im = Image.fromarray(im.numpy(), mode='L')
    imsize = min(im.size)

    if patchsize >= imsize:
        return im, (None, None)

    if xy:
        x, y = xy
        box = (x, y, x+patchsize, y+patchsize)
        patch = im.crop(box)
    else:
        x = rng.integers(0, im.size[0] - patchsize)
        y = rng.integers(0, im.size[1] - patchsize)
        box = (x, y, x+patchsize, y+patchsize)
        patch = im.crop(box)

    return pil_to_tensor(patch).squeeze()


def _get_shadow_mask(mask, up_px):

    ind = mask.nonzero()
    # Shift mask vertically up by `up_px` pixels
    ind[:, 0] = ind[:, 0] - up_px
    ind[ind[:, 0] < 0, 0] = 0

    shifted_mask = torch.full_like(mask, fill_value=False)
    shifted_mask[ind[:, 0], ind[:, 1]] = True
    shadow = torch.logical_and(shifted_mask, ~mask)

    return shadow


def fuse(texture, digit, rng=13, shadow_px=None):
    """ Blend MNIST digit with a textured background

    Args
    ----
    texture: torch.Tensor
        Gray-scale texture image
    digit: torch.Tensor
        MNIST digit

    Returns
    -------
    patch_img: torch.Tensor
        The MNIST digit blended with the texture patch
    mask: torch.Tensor
        The mask used to do the blending
    """

    rng = utils.check_rng(rng)

    # extract patch from texture database.
    patchsize = digit.shape[0]  # assumes square image
    texture = extract_patch(texture, patchsize, rng)

    # generate binary mask for digit outline
    bool_mask = (digit > 0.1 * 255)
    mask = bool_mask.to(torch.uint8)

    # The blending
    background = (1-mask) * texture
    foreground = mask * digit
    new_img = background+foreground

    # Add shadow, if shadow_px was provided
    if shadow_px is not None:
        shadow_mask = _get_shadow_mask(bool_mask, shadow_px)
        new_img[shadow_mask] = torch.min(new_img)

    return new_img, mask
