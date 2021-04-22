import torch
import numpy as np
from torch.utils.data import Dataset
from utils.tensor_ops import normalize, im2tensor
import skimage.io as io
import glob

class ToTensor(object):
    """Converts numpy ndarrays in sample to Tensors"""
    def __call__(self, sample):   
        return {
            'base' : torch.from_numpy(sample['base']),
            'lc' : torch.from_numpy(sample['lc']),
            'lc_all': torch.from_numpy(sample['lc_all'])
        }

class Bottle128Dataset(Dataset):
    def __init__(self, dataset_root, phase='train', transform=None):
        self.base_paths = glob.glob(f'{dataset_root}/{phase}/*/base.jpg')
        self.lc_paths = glob.glob(f'{dataset_root}/{phase}/*/lc_*.jpg')
        self.num_sample = len(glob.glob(f'{dataset_root}/{phase}/*'))
        assert len(self.lc_paths) % self.num_sample == 0, 'every folder should contains same amount of lighting conditions'
        self.num_lc = len(self.lc_paths) // self.num_sample
        self.phase = phase
        self.transform = transform

    def __getitem__(self, index):
        base_idx = index // self.num_lc
        rand_shape_idx = np.random.randint(low=0, high=self.num_sample) * self.num_lc + index % self.num_lc
        base_path, lc_path, rand_shape_path = self.base_paths[base_idx], self.lc_paths[index], self.lc_paths[rand_shape_idx]
        base, lc, rand_shape = io.imread(base_path), io.imread(lc_path), io.imread(rand_shape_path)
        imgs = [base, lc, rand_shape]
        imgs = [img[:, :, None] for img in imgs]
        if self.phase == 'train' and self.transform is not None:
            imgs = self.transform(imgs)
        imgs = [normalize(im2tensor(img)) for img in imgs]
        base, lc, rand_shape = imgs
        return {'base': base,
                'rand_lc': lc,
                'rand_shape': rand_shape}

    def __len__(self):
        return len(self.lc_paths)
