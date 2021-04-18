import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class ToTensor(object):
    """Converts numpy ndarrays in sample to Tensors"""
    def __call__(self, sample):   
        return {
            'base' : torch.from_numpy(sample['base']),
            'lc' : torch.from_numpy(sample['lc'])
        }

class Bottle128Dataset(Dataset):
    """Bottle128 Dataset"""
    def __init__(self, base_img_file, lighting_img_file, num_lighting, transform=None):
        """
        Args:
            base_img_file (string): Path to base image npy array
            lighting_img_file (string): Path to lighting images npy array
            num_lighting (integer): Number of unique lighting conditioning provided in dataset
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.base_img_array = np.load(base_img_file)
        self.lighting_array = np.load(lighting_img_file)
        self.num_lighting = num_lighting
        self.transform = transform

    def __len__(self):
        return self.lighting_array.shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'base': self.base_img_array[0, idx, :, :, :].astype(np.float32)}
        random_lc = np.random.randint(low = 0, high = self.num_lighting)
        sample['lc'] = self.lighting_array[random_lc, idx, :, :, :].astype(np.float32)
        if self.transform:
            sample = self.transform(sample)

        return sample
