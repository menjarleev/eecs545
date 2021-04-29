import matplotlib.pyplot as plt
import os

def plot_lc_feat(gt_lc, path, label, index):
    gt_lc_i = gt_lc[0, :, :, :]
    for i, gt in enumerate(gt_lc_i):
        gt = gt.detach().cpu().numpy()
        plt.subplot(1, gt_lc.shape[1], i + 1)
        plt.imshow(gt, cmap='gray')
        plt.axis("off")
    _dir = os.path.join(path, label)
    os.makedirs(_dir, exist_ok=True)
    plt.savefig(os.path.join(_dir, f'feat_{index + 1:>04}.png'))
    plt.close()
