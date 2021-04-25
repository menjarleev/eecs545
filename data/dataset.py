import torch
import numpy as np
from torch.utils.data import Dataset
from utils.tensor_ops import normalize, im2tensor
import skimage.io as io
import os
import glob


class BaseDataset(Dataset):
    def __init__(self, base_paths, lc_paths, phase='train', transform=None, use_ref=None, num_lighting_infer=9):
        self.phase = phase
        self.transform = transform
        self.use_ref = use_ref
        self.num_lighting_infer = num_lighting_infer
        self.base_paths = base_paths
        self.lc_paths = lc_paths
        self.num_sample = len(self.base_paths)
        self.num_lc = len(self.lc_paths) // self.num_sample
        assert len(self.lc_paths) % self.num_sample == 0, 'every folder should contains same amount of lighting conditions'

    def __getitem__(self, index):
        base_idx = index // self.num_lc
        rand_shape_idx = np.random.randint(low=0, high=self.num_sample) * self.num_lc + index % self.num_lc
        rand_idx = np.random.randint(low=0, high=self.num_sample)
        base_path, lc_path, rand_shape_path, rand_path = self.base_paths[base_idx], self.lc_paths[index], self.lc_paths[
            rand_shape_idx], self.base_paths[rand_idx]
        base, lc, rand_shape, rand = io.imread(base_path), io.imread(lc_path), io.imread(rand_shape_path), io.imread(
            rand_path)
        imgs = [base, lc, rand_shape, rand]
        imgs = [img[:, :, None] for img in imgs]
        if self.phase == 'train' and self.transform is not None:
            imgs = self.transform(imgs)
        imgs = [normalize(im2tensor(img)) for img in imgs]
        base, lc, rand_shape, rand = imgs
        return_dict = {'base': base,
                       'rand_lc': lc,
                       'rand_shape': rand_shape,
                       'rand': rand}
        if self.phase == 'valid' or self.phase == 'test':
            ref_lc = np.random.choice(np.arange(self.num_lc), self.num_lighting_infer, replace=False)
            ref_idx = np.random.randint(low=0, high=self.num_sample) * self.num_sample + ref_lc
            ref_path = [self.lc_paths[r_idx] for r_idx in ref_idx]
            refs = [io.imread(ref_p) for ref_p in ref_path]
            refs = [ref[:, :, None] for ref in refs]
            refs = [normalize((im2tensor(ref))) for ref in refs]
            ref_stack = np.stack(refs, axis=0)
            return_dict['ref'] = ref_stack
        return return_dict

    def __len__(self):
        return len(self.lc_paths)


class Bottle128Dataset(BaseDataset):
    def __init__(self, dataset_root, phase='train', transform=None, use_ref=False, num_lighting_infer=-1, **kwags):
        base_paths = sorted(glob.glob(f'{dataset_root}/{phase}/*/base.jpg'))
        lc_paths = sorted(glob.glob(f'{dataset_root}/{phase}/*/lc_*.jpg'))
        super(Bottle128Dataset, self).__init__(base_paths, lc_paths, phase, transform, use_ref, num_lighting_infer=num_lighting_infer)


class TrashBinDataset(BaseDataset):
    def __init__(self, dataset_root, phase='train', transform=None, use_ref=False, num_lighting_infer=-1, split=.8):
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
        base_paths = set(base_paths)
        lc_paths = set(lc_paths)
        lc_paths = sorted(list(lc_paths.difference(base_paths)))
        base_paths = sorted(list(base_paths))
        super(TrashBinDataset, self).__init__(base_paths, lc_paths, phase, transform, use_ref, num_lighting_infer=num_lighting_infer)