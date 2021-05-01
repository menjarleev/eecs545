import numpy as np
import pandas as pd
import skimage.measure
import skimage.morphology
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matlab.engine
import scipy.io
import io
import glob

lighting_order_34 = [
    29,9,20,24,19,8,34,
    28,4,16,23,15,3,33,
    27,2,14,0,13,1,32,
    26,7,18,5,17,6,31,
    25,12,22,10,21,11,30
]

lighting_order_34ext = [
    29, 9, 20, 35, 24, 42, 19, 8,  34,
    28, 4, 16, 36, 23, 43, 15, 3,  33,
    49, 50,51, 37, 52, 44, 53, 54, 55,
    27, 2, 14, 38, 0,  45, 13, 1,  32,
    56, 57,58, 39, 59, 46, 60, 61, 62,
    26, 7, 18, 40, 5,  47, 17, 6,  31,
    25, 12,22, 41, 10, 48, 21, 11, 30
]


# lighting_order = list(range(1,41)) + [0] + list(range(41,81))
# lighting_order = list(range(1,61)) + [0] + list(range(61,121))
# lighting_order = list(range(1,61)) + [0] + list(range(61,121))
# lighting_order = list(range(1,25)) + [0] + list(range(25,49))

def avg_lighting(mtrx, light_set_dims=(5,7), kernel_size=3):
    print('Kernel Size:',kernel_size)
    print('Light Dims:',light_set_dims)

    # light_set_dims = (5, 7)
    # light_set_dims = (13, 13)
    # light_set_dims = (9, 9)
    # light_set_dims = (11, 11)

    print('ld:',light_set_dims[0]*light_set_dims[1])

    if light_set_dims[0]*light_set_dims[1] == 63:
        lighting_order = lighting_order_34ext
    elif light_set_dims[0]*light_set_dims[1] == 35:
        lighting_order = lighting_order_34
    else:
        lighting_order = list(range(1,25)) + [0] + list(range(25,49))

    reordered_matrix = mtrx[...,lighting_order]
    # reordered_matrix = mtrx

    print(reordered_matrix.shape)

    reordered_matrix = reordered_matrix.reshape((reordered_matrix.shape[0],
                                                 reordered_matrix.shape[1],
                                                 light_set_dims[0],
                                                 light_set_dims[1]))

    print(reordered_matrix.shape)

    fig, ax = plt.subplots(light_set_dims[0], light_set_dims[1], figsize=(20, 20))
    for i in range(light_set_dims[0]):
        for j in range(light_set_dims[1]):
            ax[i, light_set_dims[1]-1-j].imshow(reordered_matrix[...,i,j])

            ax[i, light_set_dims[1]-1-j].axes.xaxis.set_visible(False)
            ax[i, light_set_dims[1]-1-j].axes.yaxis.set_visible(False)

    plt.show()

    # raise ValueError()

    # reordered_matrix = reordered_matrix[...,3:11,3:11]

    avg_version = np.zeros((reordered_matrix.shape[0],reordered_matrix.shape[1],
                            light_set_dims[0]-(kernel_size-1),light_set_dims[1]-(kernel_size-1)))

    for i in range(light_set_dims[0]-(kernel_size-1)):
        for j in range(light_set_dims[1]-(kernel_size-1)):
            avg_version[...,i,j] = reordered_matrix[...,i:i+(kernel_size),j:j+(kernel_size)].sum(axis=-1).sum(axis=-1)/kernel_size**2

            fig, ax = plt.subplots(kernel_size, kernel_size+1, figsize = (10, 10))
            for k in range(kernel_size):
                for l in range(kernel_size):
                    ax[k, l].imshow(reordered_matrix[...,i+(k),j+(l)])

                    ax[k, l].axes.xaxis.set_visible(False)
                    ax[k, l].axes.yaxis.set_visible(False)
            ax[kernel_size-1,kernel_size].imshow(avg_version[...,i,j])
            plt.show()

    print('HERE')

    for i in range(light_set_dims[0]-(kernel_size-1)):
        for j in range(light_set_dims[1] - (kernel_size - 1)):
            plt.imshow(avg_version[...,i,j])
            plt.show()

    ret = avg_version.reshape((reordered_matrix.shape[0],reordered_matrix.shape[1],-1))

    return ret

def infer_studio(M_lst, I_lst, L_lst, light_set_dims=(5,7)):
    # studio_set = [lighting_order[10], lighting_order[16], lighting_order[18], lighting_order[24]]
    # studio_set = [lighting_order[84], lighting_order[85], lighting_order[18], lighting_order[24]]
    # studio_set = [84, 85, 97, 72]
    # studio_set = [lighting_order[59], lighting_order[61], lighting_order[70], lighting_order[50]]
    # studio_set = [40, 41, 49, 32]

    if light_set_dims[0]*light_set_dims[1] == 63:
        lighting_order = lighting_order_34ext
    elif light_set_dims[0]*light_set_dims[1] == 35:
        lighting_order = lighting_order_34
    else:
        lighting_order = list(range(1,25)) + [0] + list(range(25,49))

    cen = len(M_lst)//2
    left = cen - 1
    right = cen + 1
    up = cen - light_set_dims[1]
    down = cen + light_set_dims[1]

    ul = up-1
    ur = up+1
    dl = down-1
    dr = down+1


    studio_set = [
                  lighting_order[left],
                  lighting_order[right],
                  lighting_order[up],
                  lighting_order[down],

                  # lighting_order[ul],
                  # lighting_order[ur],
                  # lighting_order[dl],
                  # lighting_order[dr]
                  ]

    M = np.concatenate(M_lst, -1)[...,studio_set].mean(axis=-1, keepdims=True)
    I = np.concatenate(I_lst, -1)[...,studio_set].mean(axis=-1, keepdims=True)
    L = np.concatenate(L_lst, -1)[...,studio_set].mean(axis=-1, keepdims=True)

    plt.imshow(M)
    plt.show()

    return M, I, L

def get_fn_image(fn):
    print(fn)
    tmp = np.array(cv2.imread(fn, cv2.IMREAD_GRAYSCALE)).astype(float)
    tmp = np.pad(tmp, [((128 - tmp.shape[0]) // 2, (128 - tmp.shape[0]) // 2),
                       ((128 - tmp.shape[1]) // 2, (128 - tmp.shape[1]) // 2)])
    tmp = (tmp >= 0) * tmp * 255 / tmp.max()
    tmp = np.round(tmp, 0)
    tmp = tmp.astype(np.uint8)

    return tmp

def setup_PMS_output(filenames, type=None, aug_num=0, kernel_size=3,
                     light_set_dims=(5,3), infer_stud=False):
    # Creates the concatenated mask, intensity, lighting values for the Photometric Stereo
    #
    # params:
    # * filenames: list of the filenames representing different lighting conditions for an image
    # * aug_num: int number of the augmentation in question
    # returns:
    # * M (mask): np.array — 256 x 256 x K  (True for keep, False for mask out)
    # * I (Intensities): np.array — 256 x 256 x K greyscale intensities
    # * L (Lighting Direction): np.array — Dummy K x 3 greyscale intensities (all zeros)

    M_lst = []
    I_lst = []
    L_lst = []

    # Loop through all files with different lighting conditions
    for fn in filenames:
        # print(fn)

        if type is None:
            # Load data for lighting condtion's augmentation num n
            tmp = scipy.io.loadmat(fn)['output'][aug_num, ...].astype(float)

            # Adjust data for image intensities to int in the range 0-255
            tmp = np.round((tmp >= 0) * tmp * 255 / tmp.max(), 0).astype(np.uint8)
        elif type == 'jpg':
            tmp = get_fn_image(fn)
            # print(tmp)

        # try:
        #     img = np.array(tmp)[:, :, np.newaxis, np.newaxis]
        # except:
        #     pass
        img = np.array(tmp)[:, :, np.newaxis, np.newaxis]

        # Get mask, intensity, lighting data for current lighting condition
        M, I, L = generate_Mimage(img)

        M_lst.append(M)
        I_lst.append(I)
        L_lst.append(L)

    if kernel_size > 1 and infer_stud:
        tmp_M, tmp_I, tmp_L = infer_studio(M_lst, I_lst, L_lst, light_set_dims=light_set_dims)
        M_lst = [tmp_M] + M_lst
        I_lst = [tmp_I] + I_lst
        L_lst = [tmp_L] + L_lst

    # Concatenate generated mask, intensity, and lighting data for all lighting conditions
    M = (np.concatenate(M_lst, -1).sum(-1) > 0)
    I = np.concatenate(I_lst, -1)
    L = np.concatenate(L_lst, -1)

    if kernel_size > 1:
        I = avg_lighting(I, light_set_dims=light_set_dims, kernel_size=kernel_size)
        L = L[...,list(range(I.shape[-1]))]

    plt.imshow(M)
    plt.show()

    return M, I, L

def generate_Mimage(img):
    # params:
    # * img: np.array — 256 x 256 x 1 (greyscale intensity) x K lighting conditions
    # returns:
    # * M (mask): np.array — 256 x 256 x K  (True for keep, False for mask out)
    # * I (Intensities): np.array — 256 x 256 x K greyscale intensities
    # * L (Lighting Direction): np.array — Dummy K x 3 greyscale intensities (all zeros)

    pool_size = 1
    binarize_thresh = 50

    # print(img.shape)

    # # Initializing image intensity and mask matrices
    I = img[...,0]
    mask = np.zeros((img.shape[0]//pool_size, img.shape[1]//pool_size))

    # Mask out dark sections of image
    retrn, binarized, = cv2.threshold(I, binarize_thresh, 255, cv2.THRESH_BINARY)
    binarized = skimage.morphology.remove_small_holes(binarized).astype(np.uint8)*255
    contours, _ = cv2.findContours(binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 255, -1)

    # Create Mask (M), Intensity Data (I), and Blank Lighting Vector (L)
    M = np.expand_dims(mask > 0, -1)
    I = I
    L = np.zeros((3, img.shape[-1]))

    return M, I, L

def generate_PMS_output(filename):
    # params:
    # * M (mask): np.array — 256 x 256 (True for keep, False for mask out)
    # * I (Intensities): np.array — 256 x 256 greyscale intensities
    # * L (Lighting Direction): np.array — Dummy K x 3 greyscale intensities (all zeros)
    # returns:
    # * output: np.array — N x 256 x 256 x 1 for greyscale x 2 for (0 = intensity / 1 = depth)

    eng = matlab.engine.start_matlab()

    # for image_num in range(img.shape[0]):
    #     mlb_img = generate_Mimage(M, I, L)

    out = io.StringIO()
    err = io.StringIO()

    # eng.demo_tv2(M, I, L, stdout=out, stderr=err)
    U = eng.demo_tv2(filename, stdout=out, stderr=err, nargout=1)
    print('out',out.getvalue())
    print('err',err.getvalue())
    return U, eng


def main_underlying(bottle_num, base_dir_src_img, img_num_lst, vgan=True,
                    kernel_size=3, light_set_dims=(5,7), conv_load=None,
                    infer_stud=False
                    ):

    fn_suffix = 'gan' if vgan else 'pms'
    fn_avg_suffix = 'avg' if kernel_size > 1 else 'navg'

    tmp_filename = f'./PMS_out/bottle_setup_{bottle_num}_{fn_suffix}_{fn_avg_suffix}'
    out_filename = f'./PMS_out/bottle_out_{bottle_num}_{fn_suffix}_{fn_avg_suffix}'

    filename = [base_dir_src_img+f'{img_num}.jpg' for img_num in img_num_lst]
    # filename = [f'./Test_Data_128_ActModData_Studio/{bottle_num}.jpg']+filename

    if conv_load:
        # conv_load = scipy.io.loadmat(conv_dir)['idx_alph2orig'].astype(float)-1
        # conv_lst = list(conv_load[0].astype(int))
        # filename = [fn for _,fn in sorted(zip(conv_lst, filename))]
        filename2 = list(np.array(filename)[conv_load])
        filename = filename2

    if infer_stud:
        pass
    elif bottle_num > 100:
        filename = [f'./Test_Data_128_ActModData_Studio/{bottle_num}.jpg'] + filename
    else:
        filename = [f'./Validation_Data_128_ActModData_Studio/{bottle_num}.jpg'] + filename

    # print('filename list',filename)



    M, I, L = setup_PMS_output(filename, type='jpg', aug_num=None,
                               kernel_size=kernel_size, light_set_dims=light_set_dims,
                               infer_stud=infer_stud)

    fig, ax = plt.subplots(3,3, figsize=(10,10))
    ax[0,0].imshow(I[...,0])
    ax[0,1].imshow(I[...,1])
    ax[0,2].imshow(I[...,2])
    ax[1,0].imshow(I[...,3])
    ax[1,1].imshow(I[...,4])
    ax[1,2].imshow(I[...,5])
    ax[2,0].imshow(I[...,6])
    ax[2,1].imshow(I[...,7])
    ax[2,2].imshow(I[...,8])
    plt.show()

    # plt.imshow(M)
    # plt.show()

    scipy.io.savemat(f'{tmp_filename}.mat', {'M': M, 'I':I, 'L':L})
    U, eng = generate_PMS_output(f'{tmp_filename}')
    scipy.io.savemat(f'{out_filename}.mat', {'U': U})

    return U, eng

# def main_ground(bottle_num=101, max_lighting_num=35):
#     # params:
#     # * max_lighting_num (mask): int = the largest lighting number in the filename to use
#     # returns:
#     # * U: np.array = the predicted X-dimension for the input image
#
#     min_bottle_num = 101
#     base_dir_src_img = './test_rand/fake/'
#     img_num_lst = [(bottle_num - min_bottle_num)*(max_lighting_num-1) + img_num for img_num in range(1, max_lighting_num)]
#
#     U, eng = main_underlying(bottle_num, base_dir_src_img, img_num_lst)
#
#     return U, eng

def main(bottle_num=101, light_set_dims=(5,7),
         base_dir_src_img='./test_rand/fake/', conv_dir=None,
         kernel_size=3, vgan=True, infer_stud=False):
    # def main_gan(bottle_num=101, max_lighting_num=35):
    # params:
    # * max_lighting_num (mask): int = the largest lighting number in the filename to use
    # returns:
    # * U: np.array = the predicted X-dimension for the input image

    max_lighting_num = light_set_dims[0]*light_set_dims[1]

    min_bottle_num = 101 if bottle_num > 100 else 96
    # img_num_lst = [(bottle_num - min_bottle_num)*(max_lighting_num-1) + img_num for img_num in range(1, max_lighting_num)]

    img_num_lst = [(bottle_num - min_bottle_num)*(max_lighting_num-1) + img_num for img_num in range(1, max_lighting_num)]

    # print('img_num_lst', img_num_lst)

    # reordering_by_str_vs_int = list(np.argsort([int(_) for _ in sorted([str(_) for _ in img_num_lst])]))
    # img_num_lst = list(np.array(img_num_lst)[reordering_by_str_vs_int])

    # print('img_num_lst_2', reordering_by_str_vs_int)
    # print('img_num_lst_2', img_num_lst)


    if conv_dir:
        strt = min(img_num_lst)
        nd = max(img_num_lst)

        conv_load = scipy.io.loadmat(conv_dir)['idx_alph2orig'].astype(float)
        conv_load = list(conv_load[0].astype(int)[strt-1:nd] - strt)
        # conv_load = list(conv_load[0].astype(int))[strt-1:nd]
        # filename = [fn for _,fn in sorted(zip(conv_lst, filename))]
        # filename2 = list(np.array(filename)[conv_lst])


    U, eng = main_underlying(bottle_num,
                             base_dir_src_img,
                             img_num_lst,
                             vgan=vgan,
                             kernel_size=kernel_size,
                             conv_load=conv_load,
                             light_set_dims=light_set_dims,
                             infer_stud=infer_stud)

    return U, eng

# main()

# def main(bottle_num=101, max_lighting_num=10):
#     # params:
#     # * max_lighting_num (mask): int = the largest lighting number in the filename to use
#     # returns:
#     # * U: np.array = the predicted X-dimension for the input image
#
#     tmp_filename = f'bottle_setup_{bottle_num}'
#
#     for aug_num in range(1):
#         filename = [f'./Bottle_{bottle_num}/output_{img_num}' for img_num in range(1, max_lighting_num)]
#         M, I, L = setup_PMS_output(filename, aug_num=aug_num)
#
#     scipy.io.savemat(f'{tmp_filename}.mat', {'M': M, 'I':I, 'L':L})
#     U, eng = generate_PMS_output(f'{tmp_filename}')
#
#     return U, eng
#
# main()
