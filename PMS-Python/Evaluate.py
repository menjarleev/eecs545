import matplotlib.pyplot as plt
import matlab.engine
import scipy.io

# https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
# cd matlabroot/extern/engines/python
# python setup.py install

import matlab.engine
import pandas as pd
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import plotly.express as px
from plotly.subplots import make_subplots

import PMS

from threading import Thread

def matrix_loc_to_cartesian(size, shp):
    # y_vals = np.tile(np.arange(size).reshape(1, -1), (z_size, 1)) * 1.0
    vals = np.arange(size).reshape(shp) * 1.0
    vals -= np.mean(vals)
    vals = np.expand_dims(vals, 0)

    return vals


def min_max_scl(x, axis=None):
    if axis is None:
        ret = (x - x.min()) / (x.max() - x.min())
    else:
        ret = (x - x.min(axis)) / (x.max(axis) - x.min(axis))
    return ret


def get_closest_indices(basis_pts, comparison_pts):
    # (actual_pts, scl_iv)
    nrm_lst = np.zeros(basis_pts.shape[0]) + np.inf
    inx_lst = -nrm_lst.copy()
    for i in range(basis_pts.shape[0]):
        pt_a = basis_pts[i, :]
        dist_vals = np.sqrt(np.sum((comparison_pts - pt_a) ** 2, axis=-1))

        inx_lst[i] = np.argmin(dist_vals)
        nrm_lst[i] = dist_vals[int(inx_lst[i])]
    return nrm_lst, inx_lst


def get_closest_indices_yz(basis_pts, comparison_pts):
    # (actual_pts, scl_iv)
    dist_lst = np.zeros(basis_pts.shape[0]) + np.inf
    nrm_lst = np.zeros(basis_pts.shape[0]) + np.inf

    inx_lst = -dist_lst.copy()
    for i in range(basis_pts.shape[0]):
        pt_a = basis_pts[i, :]
        dist_vals = np.sqrt(np.sum((comparison_pts - pt_a)[..., 1:] ** 2, axis=-1))
        nrm_vals = np.sqrt(np.sum((comparison_pts - pt_a)[..., 0:1] ** 2, axis=-1))

        inx_lst[i] = np.argmin(dist_vals)
        dist_lst[i] = dist_vals[int(inx_lst[i])]
        nrm_lst[i] = nrm_vals[int(inx_lst[i])]
    return dist_lst, nrm_lst, inx_lst



def get_pts_closest(inx_lst_basis, inx_lst_comparison):
    pairs_basis = (-np.ones_like(inx_lst_basis)).astype(int)
    pairs_basis[inx_lst_comparison.astype(int)] = np.arange(inx_lst_comparison.shape[0])
    pairs_basis = np.concatenate([np.expand_dims(pairs_basis, -1),
                                  np.arange(pairs_basis.shape[0]).reshape((-1, 1))], -1)
    pairs_act = pairs_basis[pairs_basis[:, 0] > 0]
    pairs_keep = pairs_act[:, 1]

    return pairs_keep, pairs_act

def one_way_filter(basis_pts, comparison_pts):

    # Find the closest points to actuals and their distances
    nrm_lst_a, inx_lst_a = get_closest_indices(basis_pts, comparison_pts)
    nrm_lst_b, inx_lst_b = get_closest_indices(comparison_pts, basis_pts)

    # Get the list of points to keep
    basis_keep, pairs_act = get_pts_closest(inx_lst_a, inx_lst_b)
    scl_keep, pairs_scl = get_pts_closest(inx_lst_b, inx_lst_a)

    basis_pts_keep = basis_pts[basis_keep]
    # basis_pts_keep[:, 1:3] = 2 * min_max_scl(basis_pts_keep[:, 1:3], axis=0) - 1

    # scl_iv_keep = comparison_pts[scl_keep]
    # # scl_iv_keep[:, 1:3] = 2 * min_max_scl(scl_iv_keep[:, 1:3], axis=0) - 1

    print('Avg. Norm Dist:', nrm_lst_a.mean())

    # return basis_pts_keep
    return basis_pts_keep #, scl_iv_keep


    # # Find the closest points to actuals and their distances
    # nrm_lst_a, inx_lst_a = get_closest_points(actual_pts, scl_iv)
    # # Find the closest points to predicteds and their distances
    # nrm_lst_b, inx_lst_b = get_closest_points(scl_iv, actual_pts)
    #
    # # Get the list of points to keep
    # act_keep, pairs_act = get_pts_closest(inx_lst_a, inx_lst_b)
    # scl_keep, pairs_scl = get_pts_closest(inx_lst_b, inx_lst_a)
    #
    # # nrm_lst_a_keep = nrm_lst_a[act_keep]
    # # nrm_lst_b_keep = nrm_lst_b[scl_keep]
    #
    # actual_pts_keep = actual_pts[act_keep]
    # scl_iv_keep = scl_iv[scl_keep]
    # scl_iv_keep[:, 1:3] = 2 * min_max_scl(scl_iv_keep[:, 1:3], axis=0) - 1
    #
    # # Find the closest points to actuals and their distances
    # nrm_lst_a, inx_lst_a = get_closest_points(actual_pts_keep, scl_iv_keep)
    # # Find the closest points to predicted and their distances
    # nrm_lst_b, inx_lst_b = get_closest_points(scl_iv_keep, actual_pts_keep)
    #
    # act_keep, pairs_act = get_pts_closest(inx_lst_a, inx_lst_b)
    # scl_keep, pairs_scl = get_pts_closest(inx_lst_b, inx_lst_a)
    #
    # actual_pts_keep = actual_pts_keep[act_keep]
    # scl_iv_keep = scl_iv_keep[scl_keep]
    #
    # print('Avg. Norm Dist:', nrm_lst_a.mean())
    #
    # return



def main(act_pts_fn='./Actual_Bottle_Pts/bottle_101_actualPts.mat',
         base_dir_src_img='./test_rand/fake/',
         conv_dir=None,
         bottle_num=96,
         # averaging=False,
         vgan=True,
         kernel_size=3,
         light_set_dims=(5,7),
         infer_stud=False
         ):

    # U, eng = PMS.main()
    # for i in range(101,106):
    #     U, eng = PMS.main(bottle_num=i, base_dir_src_img=base_dir_src_img)
    #     eng.quit()

    U, eng = PMS.main(bottle_num=bottle_num,
                      base_dir_src_img=base_dir_src_img,
                      vgan=vgan,
                      kernel_size=kernel_size,
                      conv_dir=conv_dir,
                      light_set_dims=light_set_dims,
                      infer_stud=infer_stud
                      )

    npU2 = np.array(U).astype(float)
    rscld = npU2.astype(float).round(2) - npU2.astype(float).round(2).min()

    plt.imshow(rscld)
    plt.show()

    # Max-Min Scale Depth
    scl = np.expand_dims(min_max_scl(npU2), 0)

    # Get W x Z values
    y_size, z_size = npU2.shape
    # Create 1D row vector of Y coordinates tiled along matrix rows
    y_vals = np.tile(matrix_loc_to_cartesian(y_size, (1, -1)), (z_size, 1))
    # Create 1D column vector of Z coordinates tiled along matrix columns
    z_vals = -np.tile(matrix_loc_to_cartesian(z_size, (-1, 1)), (1, y_size))

    # Append X, Y, Z matrices and convert to list of all data point X, Y, Z coordinates
    indexed_version = np.concatenate([scl, y_vals, z_vals], 0).transpose((1, 2, 0)).reshape(-1, 3)

    # Remove all points that are zero or approximately zero from consideration
    scl_iv = indexed_version[indexed_version[:, 0] > 1e-3]
    # Zero out small depth values
    scl_iv[:, 1:3] = 2 * min_max_scl(scl_iv[:, 1:3], axis=0) - 1

    plt.figure(figsize=(10, 10))
    plt.imshow(scl[0, ...])

    # Load actual points
    actual_pts = scipy.io.loadmat(act_pts_fn)['actual_pts']

    # # Divide each point by the largest absolute value in each coordinate
    # actual_pts = actual_pts / np.abs(actual_pts.max(0))
    actual_pts[:, 1:3] = 2 * min_max_scl(actual_pts[:, 1:3], axis=0) - 1

    return actual_pts, scl_iv, eng

# main(base_dir_src_img='./val_rand/fake/')

if __name__ == "__main__":
    bot = 101
    act_pts_fn = f'./Actual_Bottle_Pts/bottle_{bot}_actualPts.mat'
    #
    # rt_dir = './Data/15ConstSpac'
    # # rt_dir = './Data/121ConstTheta'
    # # rt_dir = './Data/34Orig'
    # # rt_dir = './Data/34HighDensity'
    # conv_dir = f'{rt_dir}/conversion.mat'
    # base_dir = f'{rt_dir}/photometricGAN_1_lr0.001_maxstep_135000_nres_12/test_rand'
    # main(act_pts_fn=act_pts_fn, bottle_num=bot,
    #                           conv_dir=conv_dir,
    #                           # base_dir_src_img=f'{base_dir}/fake/',
    #                           base_dir_src_img=f'{base_dir}/gt/',
    #                           kernel_size=3,
    #                           # light_set_dims=(5,7),
    #                           light_set_dims=(7,7),
    #                           infer_stud=False
    #      )

    lsd_lookup = {
        '15ConstSpac': (7, 7),
        '34Orig': (5, 7),
        '34HighDensity': (7, 9),
        '34Expanded15': (5, 7),
    }

    for gan_version in ['15ConstSpac']:#, '34Orig', '34HighDensity', '34Expanded15']:
        rt_dir = f'./Data/{gan_version}'
        conv_dir = f'{rt_dir}/conversion.mat'

        light_set_dims = lsd_lookup[gan_version]

        for bot in [101]: #range(96, 106):
            subdir = 'test_rand' if bot > 100 else 'val_rand'
            act_pts_fn = f'./Actual_Bottle_Pts/bottle_{bot}_actualPts.mat'
            ppms_base_dir = f'{rt_dir}/photometricGAN_1_lr{0.001}_maxstep_135000_nres_12'

            actual_pts_keep_pms_, scl_iv_keep_pms_, eng_pms = main(act_pts_fn=act_pts_fn,
                                                                            bottle_num=bot,
                                                                            base_dir_src_img=f'{ppms_base_dir}/{subdir}/gt/',
                                                                            kernel_size=3,
                                                                            conv_dir=conv_dir,
                                                                            light_set_dims=light_set_dims,
                                                                            infer_stud=False,
                                                                            vgan=False)
