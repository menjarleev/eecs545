import matplotlib.pyplot as plt
import os

def plot_lc_feat(gt_lc, fake_lc, path):
    gt_lc_i = gt_lc[0, :, :, :]
    fake_lc_i = fake_lc[0, :, :, :]
    for i, (gt, fake) in enumerate(zip(gt_lc_i, fake_lc_i)):
        gt = gt.detach().cpu().numpy()
        fake = fake.detach().cpu().numpy()
        plt.subplot(2, gt_lc_i.shape[0], i + 1)
        plt.imshow(gt, cmap='gray')
        plt.axis("off")
        plt.subplot(2, gt_lc_i.shape[0], i + 1 + gt_lc_i.shape[0])
        plt.imshow(fake, cmap='gray')
        plt.axis("off")
    plt.savefig(os.path.join(path, 'feat.png'), dpi=400)
    plt.close()
