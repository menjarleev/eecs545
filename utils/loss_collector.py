from model.loss import *
import torch

class LossCollector:
    def __init__(self, gpu_id, loss_terms, gan_mode, lambda_l1=0., lambda_feat=0., lambda_vgg=0.,):
        self.device = torch.device(f'cuda:{gpu_id}' if gpu_id != -1 else 'cpu')
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.loss_names_G = dict()
        self.loss_names_D = dict()
        if 'gan' in loss_terms:
            self.criterionGAN = GANLoss(gan_mode, tensor=self.tensor)
            self.loss_names_G['G_GAN'] = 0
            self.loss_names_D['D_real'] = 0
            self.loss_names_D['D_fake'] = 0
        if 'feat' in loss_terms:
            self.criterionFeat = torch.nn.L1Loss()
            self.loss_names_G['G_GAN_Feat'] = 0
        if 'l1' in loss_terms:
            self.criterionL1 = torch.nn.L1Loss()
            self.loss_names_G['L1'] = 0
        if 'vgg' in loss_terms:
            self.criterionVGG = VGGLoss().to(self.device)
            self.loss_names_G['VGG'] = 0

        self.weight = {'L1': lambda_l1,
                       'feat': lambda_feat,
                       'VGG': lambda_vgg}

    def compute_GAN_losses(self, netD, fake, gt, for_discriminator):
        pred_fake = netD(fake)
        pred_real = netD(gt)
        if for_discriminator:
            assert 'D_real' in self.loss_names_D and 'D_fake' in self.loss_names_D
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            self.loss_names_D['D_real'] += loss_D_real
            self.loss_names_D['D_fake'] += loss_D_fake

        else:
            assert 'G_GAN' in self.loss_names_G, 'G_GAN is not in loss_terms'
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            self.loss_names_G['G_GAN'] += loss_G_GAN

    def compute_feat_losses(self, netD, fake, gt):
        if self.weight['feat'] == 0:
            return
        pred_fake = netD(fake)
        pred_real = netD(gt)
        loss_G_GAN_Feat = self.GAN_matching_loss(pred_real, pred_fake)
        self.loss_names_G['G_GAN_Feat'] += loss_G_GAN_Feat


    def compute_L1_losses(self, fake, gt):
        if self.weight['L1'] == 0:
            return
        assert 'L1' in self.loss_names_G
        loss_L1 = self.criterionL1(fake, gt)
        self.loss_names_G['L1'] += loss_L1 * self.weight['L1']

    def compute_VGG_losses(self, fake_image, gt_image):
        if self.weight['VGG'] == 0:
            return
        if type(fake_image) == list:
            fake_image = fake_image[-1]
            gt_image = gt_image[-1]
        loss_G_VGG = self.criterionVGG(fake_image, gt_image)
        self.loss_names_G['VGG'] += loss_G_VGG * self.weight['VGG']

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
        for k in loss_dict.keys():
            loss_dict[k] = 0
        return losses
