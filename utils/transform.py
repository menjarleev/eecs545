import random

import numpy as np
from skimage.transform import rotate as rot
from scipy.ndimage import shift


def crop(img, psize):
    def _crop(_img, _psize, _ix, _iy):
        _img = _img[_iy:_iy + _psize, _ix:_ix + _psize, :]
        return _img
    h, w = img[0].shape[:-1]
    ix = random.randrange(0, w-psize+1)
    iy = random.randrange(0, h-psize+1)
    if type(img) == list:
        return [_crop(i, psize, ix, iy) for i in img]
    else:
        return _crop(img, psize, ix, iy)

def pixel_shift(img, shift_range=(-20, 20)):
    def _pixel_shift(_img, _vs, _hs):
        shift(_img, (_vs, _hs, 0))
        return np.clip(_img, 0, 255).astype(np.uint8)
    vshift = random.randrange(shift_range[0], shift_range[1])
    hshift = random.randrange(shift_range[0], shift_range[1])
    if type(img) == list:
        return [_pixel_shift(i, vshift, hshift) for i in img]
    else:
        return _pixel_shift(img, vshift, hshift)


def flip(img):
    def _flip(_img, _hflip, _vflip):
        if _hflip:
            _img = _img[:, ::-1, :]
        if _vflip:
            _img = _img[::-1, :, :]
        return _img
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    if type(img) == list:
        return [_flip(i, hflip, vflip) for i in img]
    else:
        return _flip(img, hflip, vflip)

def rotate(img):
    def _rot(_img, _degree):
        return rot(_img, _degree, preserve_range=True)
    degree = random.randrange(0, 359)
    if type(img) == list:
        return [_rot(i, degree) for i in img]
    else:
        return _rot(img, degree)

class Transform:
    def __init__(self, transforms: list = []):
        self.transform = transforms

    def add_transform(self, transform):
        self.transform += [transform]

    def __call__(self, img):
        for trans in self.transform:
            img = trans(img)
        return img