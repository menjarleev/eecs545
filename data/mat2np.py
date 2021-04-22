import scipy.io
import argparse
from skimage.io import imsave
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

n_lighting = 9

def mat2np():
    base_img_tensor = scipy.io.loadmat('Test_Data_128/input.mat')
    base_img_tensor = np.expand_dims(base_img_tensor['input'], 0)

    lighting_tensor = None
    sets_v = list()
    for i in range(1, n_lighting + 1):
        sets_v.append(np.expand_dims(scipy.io.loadmat('Test_Data_128/output_' + str(i) + '.mat')['output'], 0))
    lighting_tensor = np.concatenate(sets_v)

    cs_base = base_img_tensor.shape
    cs_lighting = lighting_tensor.shape

    base_img_tensor = base_img_tensor.reshape(1, -1, cs_base[3], cs_base[4])
    lighting_tensor = lighting_tensor.reshape(n_lighting, -1, cs_lighting[3], cs_lighting[4])

    base_img_tensor = base_img_tensor*(base_img_tensor > 0)
    lighting_tensor = lighting_tensor*(lighting_tensor > 0)

    base_img_tensor = 2*(base_img_tensor / np.amax(base_img_tensor, axis=(2,3), keepdims=True))-1
    lighting_tensor = 2*(lighting_tensor / np.amax(lighting_tensor, axis=(2,3), keepdims=True))-1

    base_img_tensor = np.expand_dims(base_img_tensor, 2)
    lighting_tensor = np.expand_dims(lighting_tensor, 2)

    # Indices: Lighting Condition, Augmented Bottle Example, Channel, Pixels, Pixels
    print(base_img_tensor.shape)
    print(lighting_tensor.shape)

    np.save('test_base_img_arr.npy', base_img_tensor)
    np.save('test_lighting_arr.npy', lighting_tensor)

def mat2im(source, target):
    os.makedirs(target)
    dirs = os.listdir(source)
    dir_count = 1
    for d in dirs:
        d_base = d.split('_')[0]
        new_dir = os.path.join(target, f'{d_base}_{dir_count:>04}')
        os.makedirs(new_dir, exist_ok=True)
        input = sorted(glob.glob(f'{source}/{d}/input.mat'))
        lc = sorted(glob.glob(f'{source}//{d}/output_*.mat'))
        for i in input:
            img = scipy.io.loadmat(i)['input'][0, :, :]
            img = np.clip(img, 0, 255)
            img = img.astype(np.uint8)
            new_file = os.path.join(new_dir, 'base.jpg')
            imsave(new_file, img)
        count = 1
        for i in lc:
            img = scipy.io.loadmat(i)['output'][0, :, :]
            img = np.clip(img, 0, 255)
            img = img.astype(np.uint8)
            new_file = os.path.join(new_dir, f'lc_{count:>04}.jpg')
            imsave(new_file, img)
            count += 1
        dir_count += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default='/home/ubuntu/data')
    parser.add_argument('--mat_dir', type=str, default='BottleData')
    parser.add_argument('--img_dir', type=str, default='BottleImg')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    for phase in ['train', 'valid', 'test']:
        mat_dir = os.path.join(args.dataset_root, args.mat_dir)
        img_dir = os.path.join(args.dataset_root, args.img_dir)
        old_dir = [os.path.join(mat_dir, d) for d in os.listdir(mat_dir)]
        old_phase = list(filter(lambda x: phase in x.lower(), old_dir))[0]
        mat_phase_dir = os.path.join(mat_dir, old_phase)
        img_phase_dir = os.path.join(img_dir, phase)
        mat2im(mat_phase_dir, img_phase_dir)

