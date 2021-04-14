import os
import time
import skimage.io as io
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import generate_dataloader
from utils import LossCollector
from subprocess import call
from tqdm import tqdm, trange
import numpy as np
import glob
from util.tensor_process import tensor2im
from itertools import chain
import json


class Solver:
    def __init__(self, rand_G, studio_G=None, studio_D=None, name='PhotometricGAN'):
        self.name = name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rand_G = rand_G.to(self.device)
        self.studio_G, self.studio_D = studio_G, studio_D
        self.module_dict = {'netG': self.netG}
        self.t1, self.t2 = None, None

    def fit(self,
            gpu_id,
            ckpt_root,
            model_dir,
            lr,
            max_steps,
            batch_size,
            gamma,
            decay,
            loss_terms,
            gan_mode,
            lambda_l1,
            lambda_feat,
            lambda_vgg,
            visualizer,
            gclip=0,
            step_label='latest',
            validation=False,
            validation_interval=1000,
            continue_train=False,):
        device = t.device(f'cuda:{gpu_id}' if gpu_id != -1 else 'cpu')

        if continue_train:
            save_dir = os.path.join(ckpt_root, model_dir)
        else:
            files = glob.glob(os.path.join(ckpt_root, f'{self.name}_\d+'))
            save_dir = os.path.join(ckpt_root, f'{self.name}_{len(files)}')
            os.makedirs(save_dir)

        loss_collector = LossCollector(gpu_id=gpu_id,
                                       loss_terms=loss_terms,
                                       gan_mode=gan_mode,
                                       lambda_l1=lambda_l1,
                                       lambda_feat=lambda_feat,
                                       lambda_vgg=lambda_vgg)
        optimG = torch.optim.Adam(
            chain(self.rand_G.parameters(), self.studio_G.parameters()),
            lr,
            betas=(0.9, 0.999), eps=1e-8
        )

        schedulerG = torch.optim.lr_scheduler.MultiStepLR(
            optimG, [1000*int(d) for d in decay.split('-')],
            gamma=gamma,
        )

        optimD = torch.optim.Adam(
            self.studio_D.parameters(), lr,
            betas=(0.9, 0.999), eps=1e-8
        )
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(
            optimD, [1000*int(d) for d in decay.split('-')],
            gamma=gamma,
        )
        visualizer.log_print("# params of rand_G: {}".format(sum(map(lambda x: x.numel(), self.rand_G.parameters()))))
        visualizer.log_print("# params of studio_G: {}".format(sum(map(lambda x: x.numel(), self.studio_G.parameters()))))
        visualizer.log_print("# params of studio_D: {}".format(sum(map(lambda x: x.numel(), self.studio_D.parameters()))))

        # create dataloader
        # TODO: Aabhaas
        # NOTE: some args for generate_dataloader shall be filled
        train_dataloader = generate_dataloader()
        if validation:
            # TODO: Aabhaas
            # generate validation dataloader
            validation_dataloader = generate_dataloader()

        t1 = time.time()
        start = 0

        if continue_train:
            self.load(gpu_id, save_dir, step_label, visualizer)
            json_path = os.path.join(save_dir, 'status.txt')
            if os.path.isfile(json_path):
                with open(json_path) as json_file:
                    data = json.load(json_file)
                # update self.data
                best = data['best']
                latest = data['latest']
                visualizer.log_print('========== Resuming from iteration {}K ========'
                                     .format(latest['step'] // 1000))
                start = latest['step'] if step_label == 'latest' else best['step']
            else:
                raise FileNotFoundError('iteration file at %s is not found' % json_path)

        for step in tqdm(range(start, max_steps), desc='train', leave=False):
            try:
                inputs = next(iters)
            except (UnboundLocalError, StopIteration):
                iters = iter(train_dataloader)
                inputs = next(iters)
            rand_img = inputs[1].to(device)
            studio_img = inputs[0].to(device)

            fake_studio_img_forward, light_vec_forward =  self.studio_G(rand_img)
            fake_rand_img_forward = self.rand_G(fake_studio_img_forward, light_vec_forward)

            light_vec_backward = torch.rand_like(light_vec_forward)
            fake_rand_img_backward = self.rand_G(studio_img, light_vec_backward)
            fake_studio_img_backward, fake_light_vec_backward = self.studio_G(fake_rand_img_backward)

            # TODO: lingfei - add log dict here
            loss_collector.compute_GAN_losses(self.studio_D, fake_studio_img_backward, studio_img, for_discriminator=False)
            loss_collector.compute_VGG_losses(fake_studio_img_backward, studio_img)
            loss_collector.compute_feat_losses(self.studio_D, fake_studio_img_backward, studio_img)
            loss_collector.compute_L1_losses(fake_rand_img_forward, rand_img)
            loss_collector.compute_L1_losses(fake_light_vec_backward, light_vec_backward)

            loss_collector.loss_backward(loss_collector.loss_names_G, optimG, schedulerG)

            if gclip > 0:
                torch.nn.utils.clip_grad_value_(self.rand_G.parameters(), gclip)
                torch.nn.utils.clip_grad_value_(self.studio_G.parameters(), gclip)

            loss_collector.compute_GAN_losses(self.studio_D, fake_studio_img_backward.detach(), studio_img,for_discriminator=True)
            loss_collector.loss_backward(loss_collector.loss_names_D, optimD, schedulerD)

            loss_dict = {**self.loss_collector.loss_names_G, **self.loss_collector.loss_names_D}

            if (validation and (step + 1) % validation_interval == 0):
                self.summary_and_save(step, loss_dict)

            # if (not opt.no_test and (step + 1) % opt.test_steps == 0) or opt.debug:
            #     self.test_and_save(step)
    def test(self):
        if not opt.no_test:
            self.test_dir = self.save_dir
            os.makedirs(self.test_dir, exist_ok=True)
            self.test_loader = generate_loader(opt, 'test')
            self.test_dict = {'netG': self.netG}
            try:
                test = self.data[opt.test_name]
                Visualizer.log_print(opt, '========== [{}] current best psnr {:.2f} ssim {:.4f} score {:.2f} @ step {}K '
                                     .format(opt.test_name, test['psnr'], test['ssim'], test['score'], test['step'] // 1000))
            except:
                Visualizer.log_print(opt, 'test result not found')
                self.data[opt.test_name] = {
                    'psnr': 0,
                    'ssim': 0,
                    'score': -10,
                    'step': 0,
                }

    def test_and_save(self, step):
        opt = self.opt
        step = step + 1
        psnr, ssim = self.evaluate('test')
        score = util.calculate_score(psnr, ssim)
        test = self.data[opt.test_name]
        current_test = {
            'psnr': psnr,
            'ssim': ssim,
            'score': score,
            'step': step
        }
        if util.evaluate(current_test, test, opt.eval_metric):
            self.data[opt.test_name] = current_test
            self.save(opt.test_name, self.test_dict)
            self.save_log_iter(opt.test_name)
        test = self.data[opt.test_name]
        msg = '[test {}] psnr {:.2f} ssim {:.4f} score {:.4f} (best psnr {:.2f} ssim {:.4f} score {:.4f} @ step {}K)'.format(
            opt.test_name, psnr, ssim, score, test['psnr'], test['ssim'], test['score'], test['step'] // 1000
        )
        Visualizer.log_print(opt, msg)


    def summary_and_save(self, step, loss_dict):
        opt = self.opt
        if opt.print_mem:
            call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        self.t2 = time.time()
        errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
        err_msg = ""
        for k, v in errors.items():
            if v != 0:
                err_msg += '%s: %.3f ' % (k, v)
        curr_lr = self.schedulerG.get_lr()[0]
        step, max_steps = step + 1, self.opt.max_steps
        eta = (self.t2 - self.t1) / opt.eval_steps * (max_steps - step) / 3600

        psnr, ssim = self.evaluate('validation')
        score = util.calculate_score(psnr, ssim)
        current_latest = {
            'psnr': psnr,
            'ssim': ssim,
            'score': score,
            'step': step
        }
        best = self.data['best']
        if util.evaluate(current_latest, best, opt.eval_metric):
            self.data['best'] = current_latest
            self.save('best', self.module_dict)
            self.save_log_iter('best')
        best = self.data['best']
        self.data['latest'] = current_latest
        message = '[{}K/{}K] psnr: {:.2f} ssim: {:.4f} score: {:.4f} (best psnr: {:.2f} ssim: {:.4f} score: {:.4f} @ {}K step) \n' \
                  '{}\n' \
                  'LR:{}, ETA:{:.1f} hours'.format(step // 1000, max_steps // 1000, psnr, ssim, score,
                                                   best['psnr'], best['ssim'], best['score'],
                                                   best['step'] // 1000, err_msg, curr_lr, eta)
        Visualizer.log_print(opt, message)
        self.save('latest', self.module_dict)
        self.save_log_iter('latest')
        self.t1 = time.time()

    @torch.no_grad()
    def inference(self, data_loader, dataset_name):
        opt = self.opt
        scale = opt.scale
        method = opt.name
        if opt.save_result:
            save_root = os.path.join(self.save_dir, 'SR', opt.degradation, method, dataset_name, 'x{}'.format(scale))
            os.makedirs(save_root, exist_ok=True)
        tqdm_data_loader = tqdm(data_loader, desc='infer', leave=False)
        for i, input in enumerate(tqdm_data_loader):
            LR = input[0].to(self.device)
            path = input[1][0]
            file_name = os.path.basename(path).replace('LR{}'.format(opt.degradation), method)
            name = os.path.basename(input[1][0]).split('_')[0]
            print('process image [{}]'.format(name))
            if 'cutblur' in opt.augs:
                scale = opt.scale
                LR = F.interpolate(LR, scale_factor=scale, mode='nearest')
            SR = self.netG(LR).detach()
            SR = util.quantize(SR, 1)
            SR = tensor2im(SR, normalize=opt.normalize)

            if opt.save_result:
                save_path = os.path.join(save_root, file_name)
                io.imsave(save_path, SR)


    @torch.no_grad()
    def evaluate(self, phase):
        opt = self.opt
        method = opt.name
        scale = opt.scale
        if phase == 'test':
            dataset_name = opt.test_name
            data_loader = self.test_loader
        elif phase == 'validation':
            dataset_name = opt.train_name
            data_loader = self.validation_loader
        else:
            raise NotImplementedError('[%s] is not supported in evaluation' % phase)
        self.netG.eval()

        if opt.save_result:
            save_root = os.path.join(self.save_dir, 'SR', opt.degradation, dataset_name, 'x{}'.format(scale))
            os.makedirs(save_root, exist_ok=True)

        psnr = 0
        ssim = 0
        tqdm_data_loader = tqdm(data_loader, desc=phase, leave=False)
        for i, inputs in enumerate(tqdm_data_loader):
            HR = inputs[0].to(self.device)
            LR = inputs[1].to(self.device)
            path = inputs[2][0]

            file_name = os.path.basename(path).replace('HR', method)
            if 'cutblur' in opt.augs and HR.size() != LR.size():
                scale = HR.size(2) // LR.size(2)
                LR = F.interpolate(LR, scale_factor=scale, mode='nearest')

            SR = self.netG(LR).detach()
            SR = util.quantize(SR, 1)
            HR, SR = tensor2im([HR, SR], normalize=opt.normalize)
            if opt.save_result:
                save_path = os.path.join(save_root, file_name)
                io.imsave(save_path, SR)
            crop_size = opt.scale + (6 if phase == 'validation' else 0)
            HR = HR[crop_size:-crop_size, crop_size:-crop_size, :]
            SR = SR[crop_size:-crop_size, crop_size:-crop_size, :]
            HR, SR = util.rgb2ycbcr(HR), util.rgb2ycbcr(SR)
            psnr += util.calculate_psnr(HR, SR)
            ssim += structural_similarity(HR, SR, data_range=255, multichannel=False, gaussian_weights=True, K1=0.01, K2=0.03)

        self.netG.train()

        return psnr/len(data_loader), ssim/len(data_loader)

    def save_log_iter(self, label):
        json_path = os.path.join(self.save_dir, 'status.txt')
        with open(json_path, 'w') as json_file:
            json.dump(self.data, json_file)
        Visualizer.log_print(self.opt, 'update [{}] for status file'.format(label))

    def save(self, step_label, module_dict):
        def update_state_dict(dict):
            for k, v in dict.items():
                if 'net' in k:
                    state_dict.update({k: v.cpu().state_dict()})
                    if torch.cuda.is_available():
                        v.to(self.device)
                else:
                    state_dict.update({k: v.state_dict()})
        state_dict = dict()
        update_state_dict(module_dict)
        state_path = os.path.join(self.save_dir, 'state_{}.pth'.format(step_label))
        torch.save(state_dict, state_path)

    def load(self, gpu_id, save_dir, label, visualizer):
        def load_network(network, pretrained_dict, name):
            try:
                network.load_state_dict(pretrained_dict)
                visualizer.log_print('network %s loaded' % name)
            except:
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    visualizer.log_print('Pretrained network %s has excessive layers; Only loading layers that are used' % name)
                except:
                    visualizer.log_print('Pretrained network %s has fewer layers; The following are not initialized:' % name)
                    not_initialized = set()
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add('.'.join(k.split('.')[:2]))
                    visualizer.log_print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

        state_path = os.path.join(save_dir, 'state_{}.pth'.format(label))

        if not os.path.isfile(state_path):
            visualizer.log_print('state file store in %s is not found' % state_path)
            return

        state = torch.load(state_path)

        if self.rand_G is not None:
            load_network(self.rand_G, state['rand_G'], 'rand_G')
            if gpu_id != -1:
                self.rand_G.to(f'cuda:{gpu_id}')

        if self.studio_G is not None:
            load_network(self.studio_G, state['studio_G'], 'studio_G')
            if gpu_id != -1:
                self.studio_G.to(f'cuda:{gpu_id}')

        if self.studio_D is not None:
            load_network(self.studio_D, state['studio_D], 'studio_D')
            if gpu_id != -1:
                self.studio_D.to(f'cuda:{gpu_id}')

