import json
import torch
from option.option import get_option
from torchvision import transforms

from solver import Solver
from data import Bottle128Dataset, ToTensor
from torch.utils.data import DataLoader
from model import StudioLightGenerator, RandomLightGenerator, MultiScaleDiscriminator
import numpy as np
import random
import os
from utils import LossCollector, Visualizer
from itertools import chain
import glob


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)

def main():
    opt = get_option()
    set_seed(opt.seed)
    if opt.gpu_id != -1:
        torch.backends.cudnn.benchmark = True
    # set up visualizer
    visualizer = Visualizer(opt.save_dir)
    if opt.train:
        file = os.path.join(opt.save_dir, 'opt.txt')
        args = vars(opt)
        print('--------------- Options ---------------')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print('----------------- End -----------------')
        with open(file, 'w') as json_file:
            json.dump(args, json_file)


    rand_G = RandomLightGenerator(input_dim=opt.input_dim,
                                  output_dim=opt.output_dim,
                                  z_dim=opt.z_dim,
                                  num_downsample=opt.num_downsample,
                                  num_resblock=opt.num_resblock,
                                  ngf=opt.ngf,
                                  padding_mode=opt.padding_mode_G,
                                  latent_size=opt.latent_size,
                                  max_channel=opt.max_channel)
    studio_G = StudioLightGenerator(input_dim=opt.input_dim,
                                    output_dim=opt.output_dim,
                                    z_dim=opt.z_dim,
                                    num_downsample=opt.num_downsample,
                                    num_resblock=opt.num_resblock,
                                    ngf=opt.ngf,
                                    padding_mode=opt.padding_mode_G,
                                    latent_size=opt.latent_size,
                                    max_channel=opt.max_channel)
    studio_D = None
    train_dataloader, val_dataloader = None, None
    if opt.train:
        studio_D = MultiScaleDiscriminator(input_nc=opt.input_dim,
                                           num_D=opt.num_D,
                                           n_layer=opt.n_layer_D,
                                           ndf=opt.ndf,
                                           padding_mode=opt.padding_mode_D)
        # create dataloader
        num_lighting = opt.num_lighting
        base_train = os.path.join(opt.dataset_root, 'training_base_img_arr.npy')
        lighting_train = os.path.join(opt.dataset_root, 'training_lighting_arr.npy')
        bottles_train = Bottle128Dataset(base_img_file=base_train,
                                         lighting_img_file=lighting_train,
                                         num_lighting=num_lighting,
                                         transform=transforms.Compose([ToTensor()]))
        train_dataloader = DataLoader(bottles_train, batch_size=opt.batch_size, shuffle=True)
        if opt.validation:
            base_val= os.path.join(opt.dataset_root, 'val_base_img_arr.npy')
            lighting_val= os.path.join(opt.dataset_root, 'val_lighting_arr.npy')
            bottles_val = Bottle128Dataset(base_img_file=base_val,
                                           lighting_img_file=lighting_val,
                                           num_lighting=num_lighting,
                                           transform=transforms.Compose([ToTensor()]))
            val_dataloader = DataLoader(bottles_val, batch_size=opt.batch_size_eval, shuffle=False)

    solver = Solver(rand_G, studio_G, studio_D, gpu_id=opt.gpu_id)
    if opt.train:

        # for debug
        # from pyinstrument import Profiler
        # profiler = Profiler()
        # profiler.start()
        loss_collector = LossCollector(gpu_id=opt.gpu_id,
                                       loss_terms=opt.loss_terms,
                                       gan_mode=opt.gan_mode,
                                       lambda_L1=opt.lambda_L1,
                                       lambda_feat=opt.lambda_feat,
                                       lambda_vgg=opt.lambda_vgg,
                                       lambda_vec=opt.lambda_vec,
                                       lambda_mask=opt.lambda_mask,
                                       threshold_mask=opt.threshold_mask)
        solver.fit(gpu_id=opt.gpu_id,
                   lr=opt.lr,
                   save_dir=opt.save_dir,
                   max_step=opt.max_step,
                   gamma=opt.gamma,
                   decay=opt.decay,
                   loss_collector=loss_collector,
                   visualizer=visualizer,
                   step_label=opt.step_label,
                   train_dataloader=train_dataloader,
                   val_dataloader=val_dataloader,
                   validation_interval=opt.validation_interval,
                   save_interval=opt.save_interval,
                   optim_name=opt.optim_name,
                   log_interval=opt.log_interval,
                   continue_train=opt.continue_train,
                   save_result=opt.save_result)
        # profiler.stop()
        # print(profiler.output_text())
    if opt.test:
        # create dataloader
        # substitute train with test dataset you like
        num_lighting = opt.num_lighting
        base_train = os.path.join(opt.dataset_root, 'training_base_img_arr.npy')
        lighting_train = os.path.join(opt.dataset_root, 'training_lighting_arr.npy')
        bottles_train = Bottle128Dataset(base_img_file=base_train,
                                         lighting_img_file=lighting_train,
                                         num_lighting=num_lighting,
                                         transform=transforms.Compose([ToTensor()]))
        train_dataloader = DataLoader(bottles_train, batch_size=opt.batch_size_eval, shuffle=False)
        solver.test(gpu_id=opt.gpu_id,
                    save_dir=opt.save_dir,
                    visualizer=visualizer,
                    test_dataloader=train_dataloader,
                    save_result=True,
                    step_label=opt.step_label,
                    test_step=opt.test_step)

    if opt.inference:
        # create dataloader
        # substitute train with infer dataset you like
        base_val = os.path.join(opt.dataset_root, 'val_base_img_arr.npy')
        lighting_val = os.path.join(opt.dataset_root, 'val_lighting_arr.npy')
        bottles_val = Bottle128Dataset(base_img_file=base_val,
                                       lighting_img_file=lighting_val,
                                       num_lighting=opt.num_lighting,
                                       transform=transforms.Compose([ToTensor()]))
        val_dataloader = DataLoader(bottles_val, batch_size=opt.batch_size_eval, shuffle=False)
        solver.inference(gpu_id=opt.gpu_id,
                         dataloader=val_dataloader,
                         save_dir=opt.save_dir,
                         noise_nc=opt.noise_nc,
                         input_size=opt.input_size,
                         batch_size=opt.batch_size_eval,
                         num_lighting_infer=opt.num_lighting_infer,
                         label=opt.label_infer,
                         visualizer=visualizer)


if __name__ == '__main__':
    main()

