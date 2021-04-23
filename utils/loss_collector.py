from model.loss import *
import torch

class LossCollector:
    def __init__(self,
                 gpu_id,
                 loss_terms,
                 gan_mode,
                 lambda_L1=0.,
                 lambda_feat=0.,
                 lambda_vgg=0.,
                 lambda_vae=0.,
                 lambda_kl=1.):
        self.device = torch.device(f'cuda:{gpu_id}' if gpu_id != -1 else 'cpu')
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.loss_names_G = dict()
        self.loss_names_D = dict()
        self.weight = {}
        if 'gan' in loss_terms:
            self.criterionGAN = GANLoss(gan_mode, tensor=self.tensor)
        if 'feat' in loss_terms:
            self.criterionFeat = torch.nn.L1Loss()
            self.weight['feat'] = lambda_feat
        if 'l1' in loss_terms:
            self.criterionL1 = torch.nn.L1Loss()
            self.weight['L1'] = lambda_L1
        if 'vgg' in loss_terms:
            self.criterionVGG = VGGLoss().to(self.device)
            self.weight['VGG'] = lambda_vgg
        if 'vae' in loss_terms:
            self.weight['kl'] = lambda_kl
            self.weight['vae'] =lambda_vae


    def compute_GAN_losses(self, netD, fake, gt, for_discriminator, cls):
        pred_fake = netD(fake)
        pred_real = netD(gt)
        if for_discriminator:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            self.loss_names_D[f'{cls}_D_real'] = loss_D_real
            self.loss_names_D[f'{cls}_D_fake'] = loss_D_fake
        else:
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            self.loss_names_G[f'{cls}_G_GAN'] = loss_G_GAN

    def compute_feat_losses(self, netD, fake, gt, smoother: float = 1.):
        if not 'feat' in self.weight.keys():
            return
        pred_fake = netD(fake)
        pred_real = netD(gt)
        loss_G_GAN_Feat = self.GAN_matching_loss(pred_real, pred_fake)
        self.loss_names_G[f'G_GAN_Feat'] = loss_G_GAN_Feat * smoother


    def compute_L1_losses(self, fake, gt, cls='vec', smoother: float = 1.):
        if not 'L1' in self.weight.keys():
            return
        loss_L1 = self.criterionL1(fake, gt)
        self.loss_names_G[f'{cls}_L1'] = loss_L1 * self.weight['L1'] * smoother

    def compute_VAE_losses(self, recons, input, mu, log_var):
        if not 'kl' in self.weight.keys():
            return
        kld_weight = self.weight['kl'] # Account for the minibatch samples from the dataset
        recons_loss = self.criterionL1(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        self.loss_names_G['reconstruction'] = recons_loss * self.weight['vae']
        self.loss_names_G['KLD'] = kld_loss * kld_weight

    def compute_VGG_losses(self, fake_image, gt_image):
        if not 'VGG' in self.weight:
            return
        if type(fake_image) == list:
            fake_image = fake_image[-1]
            gt_image = gt_image[-1]
        loss_G_VGG = self.criterionVGG(fake_image, gt_image)
        self.loss_names_G['VGG'] = loss_G_VGG * self.weight['VGG']

    def GAN_matching_loss(self, pred_real, pred_fake):
        loss_G_GAN_Feat = 0
        num_D = len(pred_fake)
        D_masks = 1.0 / num_D
        for i in range(num_D):
            for j in range(len(pred_fake[i])-1):
                loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                loss_G_GAN_Feat = loss_G_GAN_Feat + D_masks * loss
        return loss_G_GAN_Feat * self.weight['feat']

    def loss_backward(self, loss_dict, optimizer, scheduler):
        losses = [torch.mean(v) if not isinstance(v, int) else v for _, v in loss_dict.items()]
        loss = sum(losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return losses

