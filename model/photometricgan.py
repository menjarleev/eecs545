import os
import skimage.io as io
import torch
import torch.nn.functional as F
from subprocess import call
from tqdm import tqdm, trange
from skimage.metrics import structural_similarity
from utils.utils import calculate_psnr
from utils.tensor_ops import tensor2im
from itertools import chain
import json


class PhotometricGAN(torch.nn.Module):
    def __init__(self, rand_G, lc_G, studio_G=None, rand_D=None, name='PhotometricGAN', gpu_id=-1):
        super().__init__()
        self.name = name
        self.lc_G = lc_G
        self.rand_G = rand_G
        self.rand_D = rand_D
        self.studio_G = studio_G
        self.device = torch.device(f'cuda:{gpu_id}' if gpu_id != -1 else 'cpu')

    def fit(self,
            gpu_id,
            lr,
            save_dir,
            max_step,
            finetune_step,
            gamma,
            decay,
            loss_collector,
            visualizer,
            step_label='latest',
            optim_name='Adam',
            finetune=True,
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
        optimG = create_optimizer(chain(self.rand_G.parameters(), self.studio_G.parameters(), self.lc_G.parameters()),
                                  optim_name)
        optimD = create_optimizer(chain(self.rand_D.parameters(), ), optim_name)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(
            optimG, [1000 * int(d) for d in decay.split('-')],
            gamma=gamma,
        )

        schedulerD = torch.optim.lr_scheduler.MultiStepLR(
            optimD, [1000 * int(d) for d in decay.split('-')],
            gamma=gamma,
        )
        visualizer.log_print("# params of lc_G: {}".format(sum(map(lambda x: x.numel(), self.lc_G.parameters()))))
        visualizer.log_print("# params of rand_G: {}".format(sum(map(lambda x: x.numel(), self.rand_G.parameters()))))
        visualizer.log_print("# params of rand_D: {}".format(sum(map(lambda x: x.numel(), self.rand_D.parameters()))))
        visualizer.log_print("# params of studio_G: {}".format(sum(map(lambda x: x.numel(), self.studio_G.parameters()))))

        start = 0

        if continue_train:
            self.load(save_dir, step_label, visualizer)
            json_path = os.path.join(save_dir, 'result.txt')
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
        begin_finetune = False
        for step in tqdm(range(start, max_step), desc='train', leave=False):
            try:
                inputs = next(iters)
            except (UnboundLocalError, StopIteration):
                iters = iter(train_dataloader)
                inputs = next(iters)
            if finetune and (step + 1) >= finetune_step and not begin_finetune:
                begin_finetune = True
                self.studio_G.eval()
                self.rand_G.eval()
                for _, param in self.studio_G.named_parameters():
                    param.requires_grad = False
                for _, param in self.rand_G.named_parameters():
                    param.requires_grad = False
            rand_lc = inputs['rand_lc'].to(self.device)
            studio = inputs['base'].to(self.device)
            rand_shape = inputs['rand_shape'].to(self.device)
            b_size = rand_lc.shape[0]

            fake_studio_fwd, lc_vec_fwd = self.studio_G(rand_lc)
            _, lc_vec_fwd_hat = self.studio_G(rand_shape)
            fake_rand_lc_fwd = self.rand_G(fake_studio_fwd, (lc_vec_fwd_hat + lc_vec_fwd) / 2)
            lc_vec_bwd = self.lc_G.sample(b_size, self.device) if (not finetune or begin_finetune) else \
                torch.randn_like(lc_vec_fwd, requires_grad=False).to(self.device)
            fake_rand_bwd = self.rand_G(studio, lc_vec_bwd)
            fake_rand_lc_bwd = self.rand_G(studio, lc_vec_bwd)
            fake_studio_bwd, fake_lc_vec_bwd = self.studio_G(fake_rand_lc_bwd)
            gan_fake = torch.cat([fake_rand_bwd, studio, F.interpolate(lc_vec_bwd, studio.shape[2:]).detach()], dim=1)
            gan_real = torch.cat([rand_lc, studio, F.interpolate(lc_vec_fwd, studio.shape[2:]).detach()], dim=1)

            def compute_finetune_loss(for_D=False):
                if not for_D:
                    loss_collector.compute_GAN_losses(self.rand_D, gan_fake, gan_real.detach(), for_discriminator=False,
                                                      cls='rand')
                    loss_collector.compute_L1_losses(fake_studio_bwd, studio, 'studio_bwd')
                    if loss_collector.has_VAE:
                        loss_collector.compute_VAE_losses(*self.lc_G(lc_vec_fwd.detach()))
                else:
                    loss_collector.compute_GAN_losses(self.rand_D, gan_fake.detach(), gan_real.detach(),
                                                      for_discriminator=True, cls='rand')

            def compute_encoding_loss():
                loss_collector.compute_L1_losses(fake_studio_fwd, studio, 'studio_fwd')
                loss_collector.compute_L1_losses(fake_rand_lc_fwd, rand_lc, 'rand_fwd')
                loss_collector.compute_L1_losses(lc_vec_fwd, lc_vec_fwd_hat, 'fake_vec')

            if finetune:
                if begin_finetune:
                    compute_finetune_loss(for_D=False)
                else:
                    compute_encoding_loss()
            else:
                compute_finetune_loss(for_D=False)
                compute_encoding_loss()
            loss_G = {k: v.data.detach().item() if not isinstance(v, int) else v for k, v in loss_collector.loss_names_G.items()}
            loss_collector.loss_backward(loss_collector.loss_names_G, optimG, schedulerG)
            loss_D = {}
            if begin_finetune or not finetune:
                compute_finetune_loss(for_D=True)
                loss_D = {k: v.data.detach().item() if not isinstance(v, int) else v for k, v in loss_collector.loss_names_D.items()}
                loss_collector.loss_backward(loss_collector.loss_names_D, optimD, schedulerD)

            if (step + 1) % log_interval == 0:
                call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
                errors = {**loss_G, **loss_D}
                err_msg = ""
                for k, v in errors.items():
                    if v != 0:
                        err_msg += '%s: %.3f ' % (k, v)
                visualizer.log_print(err_msg)

            if (step + 1) % validation_interval == 0 or (step + 1) % save_interval == 0:
                curr_lr = schedulerG.get_last_lr()
                self.summary_and_save(step, max_step, save_dir, save_result, log_result, curr_lr,
                                      val_dataloader, visualizer, proceed_val=(step + 1) % validation_interval == 0)
                if begin_finetune:
                    self.rand_G.eval()
                    self.studio_G.eval()

            # if (not opt.no_test and (step + 1) % opt.test_steps == 0) or opt.debug:
            #     self.test_and_save(step)

    @torch.no_grad()
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
        json_path = os.path.join(save_dir, 'result.txt')
        if os.path.isfile(json_path):
            with open(json_path) as json_file:
                log_result = json.load(json_file)
        else:
            log_result = {}
        test_result = self.evaluate(test_dataloader, save_dir, phase='test', save_result=save_result,
                                    eval_step=test_step)
        log_result['test'] = test_result
        test_log = 'test result \n'
        for k, v in test_result.items():
            if k != 'step':
                test_log += f'{k}: {v:.2f} '
        visualizer.log_print(test_log)
        self.save_log_iter(log_result, save_dir, 'test', visualizer)

    def summary_and_save(self, step, max_step, save_dir, save_result, log_result, curr_lr, dataloader,
                         visualizer, proceed_val=False):
        best_result = log_result['best'] if 'best' in log_result else None
        if proceed_val and dataloader is not None:
            curr_result = self.evaluate(dataloader=dataloader,
                                        save_dir=save_dir,
                                        phase='val',
                                        save_result=save_result)
            curr_result.update({'step': step + 1})
            if best_result is None or (curr_result['ssim_rand'] >= best_result['ssim_rand']):
                log_result['best'] = curr_result
                best_result = curr_result
                self.save(save_dir, 'best')
                self.save_log_iter(log_result, save_dir, 'best', visualizer)
            log_result['latest'] = curr_result
            curr_log = 'curr result \n'
            for k, v in curr_result.items():
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

    def forward(self, studio, rand=None):
        bs = studio.shape[0]
        if rand is None:
            light_vec = self.lc_G.sample(1, self.device).expand(bs, -1)
        else:
            _, light_vec = self.studio_G(rand.expand_as(studio))
        return self.rand_G(studio, light_vec)

    @torch.no_grad()
    def inference(self, gpu_id, dataloader, save_dir, num_lighting_infer, label, visualizer):
        self.to(gpu_id)
        self.load(save_dir, label, visualizer)
        self.eval()
        tqdm_data_loader = tqdm(dataloader, desc='infer', leave=False)
        rand_img_dir = os.path.join(save_dir, f'infer_rand')
        os.makedirs(rand_img_dir, exist_ok=True)
        for i, inputs in enumerate(tqdm_data_loader):
            for j in range(num_lighting_infer):
                studio_img = inputs['base'].to(self.device)
                rand_ref = inputs['ref'][0, j:j+1, :, :, :].to(self.device) if 'ref' in inputs.keys() else None
                fake_rand_img = self.forward(studio_img, rand_ref)
                fake_rand_img = tensor2im(fake_rand_img)
                for k in range(studio_img.shape[0]):
                    fake_k_lighting_j = fake_rand_img[k, :, :]
                    save_folder = os.path.join(rand_img_dir, str(k + 1))
                    os.makedirs(save_folder, exist_ok=True)
                    file_path = os.path.join(save_folder, f'{j + 1}.jpg')
                    io.imsave(file_path, fake_k_lighting_j)
        self.train()

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
            base_dir = os.path.join(save_dir, phase)
            os.makedirs(base_dir, exist_ok=True)
            studio_img_dir = os.path.join(base_dir, 'studio')
            rand_img_dir = os.path.join(base_dir, 'rand')
            infer_img_dir = os.path.join(base_dir, 'infer')
            os.makedirs(studio_img_dir, exist_ok=True)
            os.makedirs(rand_img_dir, exist_ok=True)
            os.makedirs(infer_img_dir, exist_ok=True)
        for i, inputs in enumerate(tqdm_data_loader):
            rand_img = inputs['rand_lc'].to(self.device)
            studio_img = inputs['base'].to(self.device)
            rand_shape_img = inputs['rand_shape'].to(self.device)
            b_size = rand_img.shape[0]
            lc_from_noise = self.lc_G.sample(b_size, self.device)
            infer_rand = self.rand_G(studio_img, lc_from_noise)

            fake_studio_img, light_vec_forward = self.studio_G(rand_img)
            _, light_vec_ref = self.studio_G(rand_shape_img)
            fake_rand_img = self.rand_G(studio_img, light_vec_forward)
            infer_rand = tensor2im(infer_rand)
            fake_studio = tensor2im(fake_studio_img)
            fake_rand = tensor2im(fake_rand_img)
            gt_rand = tensor2im(rand_img)
            gt_studio = tensor2im(studio_img)
            for j in range(rand_img.shape[0]):
                infer_j = infer_rand[j, :, :]
                gt_rand_j = gt_rand[j, :, :]
                gt_studio_j = gt_studio[j, :, :]
                fake_rand_j = fake_rand[j, :, :]
                fake_studio_j = fake_studio[j, :, :]
                if save_result:
                    def save_result(path, label, img):
                        _dir = os.path.join(path, label)
                        os.makedirs(_dir, exist_ok=True)
                        gt_file = os.path.join(_dir, f'{idx + 1:>04}.jpg')
                        io.imsave(gt_file, img)

                    save_result(studio_img_dir, 'gt', gt_studio_j)
                    save_result(studio_img_dir, 'fake', fake_studio_j)
                    save_result(rand_img_dir, 'gt', gt_rand_j)
                    save_result(rand_img_dir, 'fake', fake_rand_j)
                    save_result(infer_img_dir, 'infer', infer_j)
                psnr_studio += calculate_psnr(gt_studio_j, fake_studio_j)
                psnr_rand += calculate_psnr(gt_rand_j, fake_rand_j)
                ssim_studio += structural_similarity(gt_studio_j, fake_studio_j, data_range=255, multichannel=False,
                                                     gaussian_weights=True, K1=0.01, K2=0.03)
                ssim_rand += structural_similarity(gt_rand_j, fake_rand_j, data_range=255, multichannel=False,
                                                   gaussian_weights=True, K1=0.01, K2=0.03)
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
            json.dump(log_result, json_file, indent=4)
        visualizer.log_print('update [{}] for status file'.format(label))

    def save(self, save_dir, label):
        state_dict = self.cpu().state_dict()
        self.to(self.device)
        state_path = os.path.join(save_dir, 'state_{}.pth'.format(label))
        torch.save(state_dict, state_path)

    def load(self, save_dir, label, visualizer):
        def load_network(network, pretrained_dict):
            try:
                network.load_state_dict(pretrained_dict)
                visualizer.log_print('network loaded')
            except:
                pretrained_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in pretrained_dict}
                    network.load_state_dict(pretrained_dict)
                    visualizer.log_print(
                        'Pretrained network has excessive layers; Only loading layers that are used')
                except:
                    visualizer.log_print(
                        'Pretrained network has fewer layers; The following are not initialized:')
                    not_initialized = set()
                    for k, v in pretrained_dict.items():
                        if v.size() == pretrained_dict[k].size():
                            pretrained_dict[k] = v

                    for k, v in pretrained_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add('.'.join(k.split('.')[:2]))
                    visualizer.log_print(sorted(not_initialized))
                    network.load_state_dict(pretrained_dict)

        state_path = os.path.join(save_dir, 'state_{}.pth'.format(label))

        if not os.path.isfile(state_path):
            visualizer.log_print('state file store in %s is not found' % state_path)
            return

        state = torch.load(state_path)
        load_network(self, state)
        self.to(self.device)
