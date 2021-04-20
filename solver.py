import os
import skimage.io as io
import torch
from subprocess import call
from tqdm import tqdm, trange
from skimage.metrics import structural_similarity
from utils import calculate_psnr, tensor2im, quantize
from itertools import chain
import json


class Solver:

    def __init__(self, rand_G, studio_G=None, studio_D=None, name='PhotometricGAN', gpu_id=-1):
        self.name = name
        self.rand_G = rand_G
        self.studio_G, self.studio_D = studio_G, studio_D
        self.device = torch.device(f'cuda:{gpu_id}' if gpu_id != -1 else 'cpu')

    def to(self, device):
        if type(device) == int:
            self.device = torch.device(f'cuda:{device}' if device != -1 else 'cpu')
        elif type(device) == torch.device:
            self.device = device
        else:
            raise ValueError(f'{device} is not a device')
        if self.rand_G is not None:
            self.rand_G.to(self.device)
        if self.studio_G is not None:
            self.studio_G.to(self.device)
        if self.studio_D is not None:
            self.studio_D.to(self.device)

    def fit(self,
            gpu_id,
            lr,
            save_dir,
            max_step,
            gamma,
            decay,
            loss_collector,
            visualizer,
            step_label='latest',
            optim_name='Adam',
            train_dataloader=None,
            val_dataloader=None,
            validation_interval=1000,
            save_interval=1000,
            log_interval=1000,
            continue_train=False,
            save_result=False):

        def create_optimizer(parameters, optim_name, **args):
            if optim_name == 'Adam':
                optimizer = torch.optim.Adam(
                    parameters,
                    lr,
                    betas=(0.9, 0.999), eps=1e-8
                )
            elif optim_name == 'SGD':
                optimizer = torch.optim.SGD(
                    parameters,
                    lr)
            else:
                raise NotImplementedError(f'{optim_name} is not implemented')
            return optimizer
        self.to(gpu_id)
        log_result = {}
        optimG = create_optimizer(chain(self.rand_G.parameters(), self.studio_G.parameters()), optim_name)
        optimD = create_optimizer(self.studio_D.parameters(), optim_name)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(
            optimG, [1000*int(d) for d in decay.split('-')],
            gamma=gamma,
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
            self.load(save_dir, step_label, visualizer)
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
                visualizer.log_print('iteration file at %s is not found' % json_path)

        for step in tqdm(range(start, max_step), desc='train', leave=False):
            try:
                inputs = next(iters)
            except (UnboundLocalError, StopIteration):
                iters = iter(train_dataloader)
                inputs = next(iters)
            rand_img = inputs['lc'].to(self.device)
            studio_img = inputs['base'].to(self.device)
            lc_img = inputs['lc_all'].to(self.device)

            fake_studio_img_forward, light_vec_forward, _ = self.studio_G(rand_img)
            fake_rand_img_forward, _ = self.rand_G(fake_studio_img_forward, light_vec_forward.detach())

            light_vec_backward = torch.rand_like(light_vec_forward, requires_grad=False)
            fake_rand_img_backward, res_rand_backward = self.rand_G(studio_img, light_vec_backward)
            fake_studio_img_backward, fake_light_vec_backward, res_studio_backward = self.studio_G(fake_rand_img_backward)

            loss_collector.compute_GAN_losses(self.studio_D, fake_studio_img_backward, studio_img, for_discriminator=False)
            segmap = torch.mean(lc_img, dim=1)
            loss_collector.compute_mask_losses(fake_rand_img_forward, fake_rand_img_backward, segmap)
            loss_collector.compute_feat_losses(self.studio_D, fake_studio_img_backward, studio_img)
            loss_collector.compute_VGG_losses(fake_studio_img_backward, studio_img)
            loss_collector.compute_L1_losses(fake_studio_img_forward, studio_img)
            loss_collector.compute_L1_losses(fake_rand_img_forward, rand_img, rand=True)
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

    def test(self,
             gpu_id,
             save_dir,
             visualizer,
             test_dataloader=None,
             save_result=True,
             step_label='best',
             test_step=-1):
        self.to(gpu_id)
        self.load(save_dir, step_label, visualizer)
        json_path = os.path.join(save_dir, 'status.txt')
        if os.path.isfile(json_path):
            with open(json_path) as json_file:
                log_result = json.load(json_file)
        else:
            log_result = {}
        test_result = self.evaluate(test_dataloader, save_dir, phase='test', save_result=save_result, eval_step=test_step)
        log_result['test'] = test_result
        self.save_log_iter(log_result, save_dir, 'test', visualizer)


    def summary_and_save(self, step, max_step, save_dir, save_result, log_result, curr_lr, dataloader, visualizer, proceed_val=False):
        best_result = log_result['best'] if 'best' in log_result else None
        if proceed_val and dataloader is not None:
            curr_result = self.evaluate(dataloader=dataloader,
                                        save_dir=save_dir,
                                        phase='val',
                                        save_result=save_result)
            curr_result.update({'step': step})
            if best_result is None or (curr_result['psnr_rand'] >= best_result['psnr_rand'] and curr_result['ssim_rand'] >= best_result['ssim_rand']):
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
            message = f'[{(step + 1) // 1000}K/{max_step // 1000}K] \n' \
                      f'lr:{curr_lr}\n' \
                      f'{curr_log}\n' \
                      f'{best_log}\n'
            visualizer.log_print(message)
            self.save_log_iter(log_result, save_dir, 'latest', visualizer)
        else:
            'save latest result'
        self.save(save_dir, 'latest')

    @torch.no_grad()
    def inference(self, gpu_id, dataloader, save_dir, noise_nc, input_size, batch_size, num_lighting_infer, label, visualizer):
        self.to(gpu_id)
        self.load(save_dir, label, visualizer)
        self.rand_G.eval()
        tqdm_data_loader = tqdm(dataloader, desc='infer', leave=False)
        rand_img_dir = os.path.join(save_dir, f'infer_rand')
        os.makedirs(rand_img_dir, exist_ok=True)
        for i, inputs in enumerate(tqdm_data_loader):
            for j in range(num_lighting_infer):
                studio_img = inputs['base'].to(self.device)
                light_vec = torch.randn((batch_size, noise_nc, *input_size)).to(self.device)
                fake_rand_img, _ = self.rand_G(studio_img, light_vec)
                fake_rand_img = tensor2im(fake_rand_img)
                for k in range(studio_img.shape[0]):
                    fake_k_lighting_j = fake_rand_img[k, :, :]
                    save_folder = os.path.join(rand_img_dir, str(k + 1))
                    os.makedirs(save_folder, exist_ok=True)
                    file_path = os.path.join(save_folder, f'{j + 1}.jpg')
                    io.imsave(file_path, fake_k_lighting_j)
        self.rand_G.train()


    @torch.no_grad()
    def evaluate(self, dataloader, save_dir, phase='test', save_result=False, eval_step=-1):
        self.rand_G.eval()
        self.studio_G.eval()
        psnr_studio = 0
        ssim_studio = 0
        psnr_rand = 0
        ssim_rand = 0
        tqdm_data_loader = tqdm(dataloader, desc=phase, leave=False)
        idx = 0
        if save_result:
            studio_img_dir = os.path.join(save_dir, f'{phase}_studio')
            rand_img_dir = os.path.join(save_dir, f'{phase}_rand')
            os.makedirs(studio_img_dir, exist_ok=True)
            os.makedirs(rand_img_dir, exist_ok=True)
        for i, inputs in enumerate(tqdm_data_loader):
            rand_img = inputs['lc'].to(self.device)
            studio_img = inputs['base'].to(self.device)

            fake_studio_img, light_vec_forward, _ = self.studio_G(rand_img)

            fake_rand_img, _ = self.rand_G(studio_img, light_vec_forward)
            crop_size = 10
            fake_studio = tensor2im(fake_studio_img)
            fake_rand = tensor2im(fake_rand_img)
            rand = tensor2im(rand_img)
            studio = tensor2im(studio_img)
            fake_studio = fake_studio[:, crop_size:-crop_size, crop_size:-crop_size]
            fake_rand = fake_rand[:, crop_size:-crop_size, crop_size:-crop_size]
            gt_studio = studio[:, crop_size:-crop_size, crop_size:-crop_size]
            gt_rand = rand[:, crop_size:-crop_size, crop_size:-crop_size]
            for j in range(rand_img.shape[0]):
                gt_rand_j = gt_rand[j, :, :]
                gt_studio_j = gt_studio[j, :, :]
                fake_rand_j = fake_rand[j, :, :]
                fake_studio_j = fake_studio[j, :, :]
                if save_result:
                    def save_result(path, gt, fake):
                        gt_dir = os.path.join(path, 'gt')
                        fake_dir = os.path.join(path, 'fake')
                        os.makedirs(gt_dir, exist_ok=True)
                        os.makedirs(fake_dir, exist_ok=True)
                        gt_file = os.path.join(gt_dir, f'{idx + 1}.jpg')
                        io.imsave(gt_file, gt)
                        fake_file = os.path.join(fake_dir, f'{idx + 1}.jpg')
                        io.imsave(fake_file, fake)
                    save_result(studio_img_dir, gt_studio_j, fake_studio_j)
                    save_result(rand_img_dir, gt_rand_j, fake_rand_j)
                psnr_studio += calculate_psnr(gt_studio_j, fake_studio_j)
                psnr_rand += calculate_psnr(gt_rand_j, fake_rand_j)
                ssim_studio += structural_similarity(gt_studio_j, fake_studio_j, data_range=255, multichannel=False, gaussian_weights=True, K1=0.01, K2=0.03)
                ssim_rand += structural_similarity(gt_rand_j, fake_rand_j, data_range=255, multichannel=False, gaussian_weights=True, K1=0.01, K2=0.03)
                idx += 1
            if eval_step != -1 and (i + 1) % eval_step == 0:
                break

        self.rand_G.train()
        self.studio_G.train()

        return {'psnr_rand': psnr_rand / idx,
                'ssim_rand': ssim_rand / idx,
                'psnr_studio': psnr_studio / idx,
                'ssim_studio': ssim_studio / idx}

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

    def load(self, save_dir, label, visualizer):
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
            self.rand_G.to(self.device)

        if self.studio_G is not None:
            load_network(self.studio_G, state['studio_G'], 'studio_G')
            self.studio_G.to(self.device)

        if self.studio_D is not None:
            load_network(self.studio_D, state['studio_D'], 'studio_D')
            self.studio_D.to(self.device)

