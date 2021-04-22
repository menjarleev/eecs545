import scipy.io
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

def mat2im(dataset_root, phase, dir_name):
    dirs = os.listdir(os.path.join(dataset_root, dir_name))

    os.makedirs(os.path.join(dataset_root, phase), exist_ok=True)
    for d in dirs:
        new_dir = os.path.join(dataset_root, phase, d)
        os.makedirs(new_dir, exist_ok=True)
        input = glob.glob(f'{dataset_root}/{dir_name}/{d}/input.mat')
        lc = glob.glob(f'{dataset_root}/{dir_name}/{d}/output_*.mat')
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
            new_file = os.path.join(new_dir, f'lc_{count}.jpg')
            imsave(new_file, img)
            count += 1


if __name__ == '__main__':
    phase = 'train'
    dataset_root = '/home/ubuntu/data/BottleData'
    dir_name = 'Train_Data_128'
    mat2im(dataset_root, phase, dir_name)
    phase = 'valid'
    dataset_root = '/home/ubuntu/data/BottleData'
    dir_name = 'Validation_Data_128'
    mat2im(dataset_root, phase, dir_name)
    phase = 'test'
    dataset_root = '/home/ubuntu/data/BottleData'
    dir_name = 'Test_Data_128'
    mat2im(dataset_root, phase, dir_name)


