import torch
import numpy as np
from torch.utils.data import Dataset
import skimage.io as io
from torchvision import transforms


class ToTensor(object):
    """Converts numpy ndarrays in sample to Tensors"""

    def __call__(self, sample):
        return {
            'base': torch.from_numpy(sample['base']),
            'lc': torch.from_numpy(sample['lc']),
            'lc_k': torch.from_numpy(sample['lc_k']),
            'ctn_k': torch.from_numpy(sample['ctn_k'])
        }


class Bottle128Dataset(Dataset):
    """Bottle128 Dataset"""

    def __init__(self, base_img_file, lighting_img_file, num_lighting, num_neg_sample=4, num_sample=95, num_aug=24, transform=None):
        """
        Args:
            base_img_file (string): Path to base image npy array
            lighting_img_file (string): Path to lighting images npy array
            num_lighting (integer): Number of unique lighting conditioning provided in dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.sample_indices= np.arange(num_sample)
        self.aug_indices = np.arange(num_aug)
        self.lighting_indices = np.arange(num_lighting)
        self.base_img_array = np.load(base_img_file).reshape(1, num_sample, num_aug, 1, 128, 128).astype(np.float32)
        self.lighting_array = np.load(lighting_img_file).reshape(num_lighting, num_sample, num_aug, 1, 128, 128).astype(np.float32)
        self.num_neg_sample = num_neg_sample
        self.transform = transform
        self.prob_sample = np.full(num_sample, 1 / (num_sample - 1))
        # self.full_indices = np.indices(self.lighting_array.shape)

    def __len__(self):
        return self.lighting_array.shape[1] * self.lighting_array.shape[2]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        aug_i = idx % len(self.aug_indices)
        sample_i = idx // len(self.aug_indices)
        sample = {'base': self.base_img_array[0, sample_i, aug_i, ...]}
        random_lc = np.random.choice(self.lighting_indices, self.num_neg_sample + 1, replace=False)
        lc_array = self.lighting_array[random_lc, sample_i, aug_i, ...]
        sample['lc'] = lc_array[0, ...].astype(np.float32)
        sample['lc_k'] = lc_array[1:, ...].astype(np.float32)

        prob_sample = self.prob_sample
        tmp_val = prob_sample[sample_i]
        prob_sample[sample_i] = 0
        rand_sample = np.random.choice(self.sample_indices, self.num_neg_sample, replace=False, p=prob_sample)
        prob_sample[sample_i] = tmp_val
        sample['ctn_k'] = self.lighting_array[random_lc[0], rand_sample, aug_i, ...]

        if self.transform:
            sample = self.transform(sample)

        return sample
