import matplotlib.pyplot as plt
import os

def plot_lc_feat(gt_lc, path, label, index):
    for n in range(gt_lc.shape[0]):
        gt_lc_i = gt_lc[n, :, :, :]
        for i, gt in enumerate(gt_lc_i):
            gt = gt.detach().cpu().numpy()
            plt.subplot(gt_lc.shape[0], gt_lc.shape[1], n * gt_lc.shape[0] + (i + 1))
            plt.imshow(gt, cmap='gray')
            plt.axis("off")
    _dir = os.path.join(path, label)
    plt.savefig(os.path.join(_dir, f'feat_{index + 1:>04}.png'), dpi=400)
    plt.close()
