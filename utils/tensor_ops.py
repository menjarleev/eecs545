import torch
import numpy as np
from torchvision.transforms import Normalize

def im2tensor(im):
    if type(im) == list:
        return [im2tensor(i) for i in im]
    else:
        np_t = np.ascontiguousarray(im.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_t).float()
        tensor = tensor.div_(255.0)
        return tensor

def tensor2im(image_tensor, range: tuple = (-1, 1)):
    if isinstance(image_tensor, list):
        return [tensor2im(t, range) for t in image_tensor]
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) - range[0]) / (range[1] - range[0]) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255).round()
    if image_numpy.shape[-1] == 1:
        image_numpy = image_numpy[:, :, :, 0]
    return image_numpy.astype(np.uint8)

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def normalize(tensor, mean=0.5, std=0.5):
    def _normalize(t):
        c, _, _ = t.size()
        if type(mean) == list:
            _mean = mean
        else:
            _mean = np.repeat(mean, c)
        if type(std) == list:
            _std = std
        else:
            _std = np.repeat(std, c)
        tensor = Normalize(_mean, _std)(t)
        return tensor

    if tensor is None:
        return None
    elif type(tensor) == list or type(tensor) == tuple:
        return [normalize(i) for i in tensor]
    else:
        return _normalize(tensor)

