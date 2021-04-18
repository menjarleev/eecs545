import os
import time
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import ToTensor, Bottle128Dataset
from utils import LossCollector
from subprocess import call
from tqdm import tqdm, trange
import numpy as np
import glob
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity
from utils import calculate_psnr, tensor2im, quantize
from itertools import chain
import json


class Solver:

    def __init__(self, rand_G, studio_G=None, studio_D=None, name='PhotometricGAN'):
        self.name = name
        self.rand_G = rand_G
        self.studio_G, self.studio_D = studio_G, studio_D

    def fit(self,
            gpu_id,
            save_dir,
            lr,
            max_step,
            gamma,
            decay,
            loss_collector,
            visualizer,
            step_label='latest',
            train_dataloader=None,
            val_dataloader=None,
            validation_interval=1000,
            save_interval=1000,
            log_interval=1000,
            continue_train=False,
            save_result=False):

        self.device = torch.device(f'cuda:{gpu_id}' if gpu_id != -1 else 'cpu')
        self.rand_G = self.rand_G.to(self.device)
        self.studio_G = self.studio_G.to(self.device)
        self.studio_D = self.studio_D.to(self.device)
        log_result = {}
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

        start = 0

        if continue_train:
            self.load(gpu_id, save_dir, step_label, visualizer)
            json_path = os.path.join(save_dir, 'status.txt')
            if os.path.isfile(json_path):
                with open(json_path) as json_file:
                    log_result = json.load(json_file)
                best_result = log_result['best']
                latest_result = log_result['latest']
                visualizer.log_print('========== Resuming from iteration {}K ========'
                                     .format(latest_result['step'] // 1000))
                start = latest_result['step'] if step_label == 'latest' else best_result['step']
            else:
                raise FileNotFoundError('iteration file at %s is not found' % json_path)

        for step in tqdm(range(start, max_step), desc='train', leave=False):
            try:
                inputs = next(iters)
            except (UnboundLocalError, StopIteration):
                iters = iter(train_dataloader)
                inputs = next(iters)
            rand_img = inputs['lc'].to(self.device)
            studio_img = inputs['base'].to(self.device)

            fake_studio_img_forward, light_vec_forward, _ = self.studio_G(rand_img)

            light_vec_backward = torch.rand_like(light_vec_forward, requires_grad=False)
            fake_rand_img_backward, res_rand_backward = self.rand_G(studio_img, light_vec_backward)
            fake_studio_img_backward, fake_light_vec_backward, res_studio_backward = self.studio_G(fake_rand_img_backward)

            # TODO: lingfei - add log dict here
            loss_collector.compute_GAN_losses(self.studio_D, fake_studio_img_backward, studio_img, for_discriminator=False)
            loss_collector.compute_feat_losses(self.studio_D, fake_studio_img_backward, studio_img)
            loss_collector.compute_VGG_losses(fake_studio_img_backward, studio_img)
            loss_collector.compute_L1_losses(fake_studio_img_forward, studio_img)
            loss_collector.compute_L1_losses(res_rand_backward, res_studio_backward, intermediate=True)
            loss_collector.compute_vec_losses(fake_light_vec_backward, light_vec_backward)

            loss_collector.loss_backward(loss_collector.loss_names_G, optimG, schedulerG)

            # if gclip > 0:
            #     torch.nn.utils.clip_grad_value_(self.rand_G.parameters(), gclip)
            #     torch.nn.utils.clip_grad_value_(self.studio_G.parameters(), gclip)

            loss_collector.compute_GAN_losses(self.studio_D, fake_studio_img_backward.detach(), studio_img, for_discriminator=True)
            loss_collector.loss_backward(loss_collector.loss_names_D, optimD, schedulerD)

            loss_dict = {**loss_collector.loss_names_G, **loss_collector.loss_names_D}

            if (step + 1) % log_interval == 0:
                call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
                errors = {k: v.data.detach().item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                err_msg = ""
                for k, v in errors.items():
                    if v != 0:
                        err_msg += '%s: %.3f ' % (k, v)
                visualizer.log_print(err_msg)

            if (step + 1) % validation_interval == 0 or (step + 1) % save_interval == 0:
                curr_lr = schedulerG.get_lr()[0]
                self.summary_and_save(step, max_step, save_dir, save_result, log_result, curr_lr, val_dataloader, visualizer, proceed_val=(step + 1) % validation_interval == 0)

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


    def summary_and_save(self, step, max_step, save_dir, save_result, log_result, curr_lr, dataloader, visualizer, proceed_val=False):
        best_result = log_result['best'] if 'best' in log_result else None
        if proceed_val and dataloader is not None:
            curr_result = self.evaluate(dataloader=dataloader,
                                        save_dir=save_dir,
                                        phase='val',
                                        save_result=save_result)
            curr_result.update({'step': step})
            if best_result is None or (curr_result['psnr_rand'] >= best_result['psnr_rand'] and curr_result['ssim_rand'] >= best_result['ssim_rand']) or (curr_result['psnr_studio'] >= best_result['psnr_studio'] and curr_result['ssim_studio'] >= best_result['ssim_studio']):
                log_result['best'] = curr_result
                best_result = curr_result
                self.save(save_dir, 'best')
                self.save_log_iter(log_result, save_dir, 'best', visualizer)
            log_result['latest'] = curr_result
            curr_log = 'curr result \n'
            for k, v in best_result.items():
                if k != 'step':
                    curr_log += f'{k}: {v:.2f} '
            best_log = 'best result \n'
            for k, v in best_result.items():
                if k != 'step':
                    best_log += f'{k}: {v:.2f} '
            message = f'[{step // 1000}K/{max_step // 1000}K] \n' \
                      f'lr:{curr_lr}' \
                      f'{curr_log}' \
                      f'{best_log}'
            visualizer.log_print(message)
            self.save_log_iter(log_result, save_dir, 'latest', visualizer)
        else:
            'save latest result'
        self.save(save_dir, 'latest')

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
            SR = quantize(SR, 1)
            SR = tensor2im(SR, normalize=opt.normalize)

            if opt.save_result:
                save_path = os.path.join(save_root, file_name)
                io.imsave(save_path, SR)


    @torch.no_grad()
    def evaluate(self, dataloader, save_dir, phase='test', save_result=False):
        self.rand_G.eval()
        self.studio_G.eval()
        psnr_studio = 0
        ssim_studio = 0
        psnr_rand = 0
        ssim_rand = 0
        tqdm_data_loader = tqdm(dataloader, desc=phase, leave=False)
        if save_result:
            studio_img_dir = os.path.join(save_dir, 'studio')
            rand_img_dir = os.path.join(save_dir, 'rand')
            os.makedirs(studio_img_dir, exist_ok=True)
            os.makedirs(rand_img_dir, exist_ok=True)
        for i, inputs in enumerate(tqdm_data_loader):
            rand_img = inputs['lc'].to(self.device)
            studio_img = inputs['base'].to(self.device)

            fake_studio_img, light_vec_forward, _ = self.studio_G(rand_img)

            fake_rand_img, _ = self.rand_G(studio_img, light_vec_forward)
            crop_size = 10
            fake_studio_img = tensor2im(quantize(fake_studio_img, 1))
            fake_rand_img = tensor2im(quantize(fake_rand_img, 1))
            rand_img = tensor2im(quantize(rand_img, 1))
            studio_img = tensor2im(quantize(studio_img, 1))
            fake_studio = fake_studio_img[crop_size:-crop_size, crop_size:-crop_size]
            fake_rand = fake_rand_img[crop_size:-crop_size, crop_size:-crop_size]
            gt_studio = studio_img[crop_size:-crop_size, crop_size:-crop_size]
            gt_rand = rand_img[crop_size:-crop_size, crop_size:-crop_size]
            if save_result:
                def save_result(path, gt, fake):
                    gt_dir = os.path.join(path, 'gt')
                    fake_dir = os.path.join(path, 'fake')
                    os.makedirs(gt_dir, exist_ok=True)
                    os.makedirs(fake_dir, exist_ok=True)
                    gt_file = os.path.join(gt_dir, f'{i + 1}.jpg')
                    io.imsave(gt_file, gt)
                    fake_file = os.path.join(fake_dir, f'{i + 1}.jpg')
                    io.imsave(fake_file, fake)
                save_result(studio_img_dir, gt_studio, fake_studio)
                save_result(rand_img_dir, gt_rand, fake_rand)
            psnr_studio += calculate_psnr(gt_studio, fake_studio)
            psnr_rand += calculate_psnr(gt_rand, fake_rand)
            ssim_studio += structural_similarity(gt_studio, fake_studio, data_range=255, multichannel=False, gaussian_weights=True, K1=0.01, K2=0.03)
            ssim_rand += structural_similarity(gt_rand, fake_rand, data_range=255, multichannel=False, gaussian_weights=True, K1=0.01, K2=0.03)

        self.rand_G.train()
        self.studio_G.train()

        return {'psnr_rand': psnr_rand / len(dataloader),
                'ssim_rand': ssim_rand / len(dataloader),
                'psnr_studio': psnr_studio / len(dataloader),
                'ssim_studio': ssim_studio / len(dataloader)}

    def save_log_iter(self, log_result, save_dir, label, visualizer):
        json_path = os.path.join(save_dir, 'result.txt')
        with open(json_path, 'w') as json_file:
            json.dump(log_result, json_file)
        visualizer.log_print('update [{}] for status file'.format(label))

    def save(self, save_dir, label):
        def update_state_dict(name, module):
            state_dict.update({name: module.cpu().state_dict()})
            module.to(self.device)
        state_dict = dict()
        update_state_dict('rand_G', self.rand_G)
        update_state_dict('studio_G', self.studio_G)
        update_state_dict('studio_D', self.studio_D)
        state_path = os.path.join(save_dir, 'state_{}.pth'.format(label))
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
            load_network(self.studio_D, state['studio_D'], 'studio_D')
            if gpu_id != -1:
                self.studio_D.to(f'cuda:{gpu_id}')

