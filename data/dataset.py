import torch
import numpy as np
from torch.utils.data import Dataset
from utils.tensor_ops import normalize, im2tensor
import skimage.io as io
import os
import glob
from abc import abstractmethod

class BaseDataset(Dataset):
    def __init__(self, phase='train', transform=None):
        self.phase = phase
        self.transform = transform

    def __getitem__(self, index):
        base_idx = index // self.num_lc
        rand_shape_idx = np.random.randint(low=0, high=self.num_sample) * self.num_lc + index % self.num_lc
        rand_idx = np.random.randint(low=0, high=self.num_sample)
        base_path, lc_path, rand_shape_path, rand_path = self.base_paths[base_idx], self.lc_paths[index], self.lc_paths[rand_shape_idx], self.base_paths[rand_idx]
        base, lc, rand_shape, rand = io.imread(base_path), io.imread(lc_path), io.imread(rand_shape_path), io.imread(rand_path)
        imgs = [base, lc, rand_shape, rand]
        imgs = [img[:, :, None] for img in imgs]
        if self.phase == 'train' and self.transform is not None:
            imgs = self.transform(imgs)
        imgs = [normalize(im2tensor(img)) for img in imgs]
        base, lc, rand_shape, rand = imgs
        return {'base': base,
                'rand_lc': lc,
                'rand_shape': rand_shape,
                'rand': rand}

    @abstractmethod
    def __len__(self):
        pass

class Bottle128Dataset(BaseDataset):
    def __init__(self, dataset_root, phase='train', transform=None):
        super(Bottle128Dataset, self).__init__(phase, transform)
        self.base_paths = sorted(glob.glob(f'{dataset_root}/{phase}/*/base.jpg'))
        self.lc_paths = sorted(glob.glob(f'{dataset_root}/{phase}/*/lc_*.jpg'))
        self.num_sample = len(glob.glob(f'{dataset_root}/{phase}/*'))
        assert len(self.lc_paths) % self.num_sample == 0, 'every folder should contains same amount of lighting conditions'
        self.num_lc = len(self.lc_paths) // self.num_sample

    def __len__(self):
        return len(self.lc_paths)

class TrashBinDataset(BaseDataset):
    def __init__(self, dataset_root, phase='train', transform=None, split=.8):
        super(TrashBinDataset, self).__init__(phase, transform)
        folder_name = sorted(os.listdir(f'{dataset_root}'))
        if phase == 'train':
            folder_name = folder_name[:int(len(folder_name) * split)]
        elif phase == 'valid':
            folder_name = folder_name[int(len(folder_name) * split):int(len(folder_name) * (split + (1 - split) / 2))]
        elif phase == 'test':
            folder_name = folder_name[int(len(folder_name) * (split + (1 - split) / 2)):]
        else:
            raise ValueError
        base_paths = []
        lc_paths = []
        for fn in folder_name:
            base_paths += glob.glob(f'{dataset_root}/{fn}/*_theta_60_phi_130*.png')
            lc_paths += glob.glob(f'{dataset_root}/{fn}/*.png')
        self.base_paths = set(base_paths)
        self.lc_paths = set(lc_paths)
        self.lc_paths = sorted(list(self.lc_paths.difference(self.base_paths)))
        self.base_paths = sorted(list(self.base_paths))
        self.num_sample = len(folder_name)
        self.num_lc = len(self.lc_paths) // self.num_sample

    def __len__(self):
        return len(self.lc_paths)
