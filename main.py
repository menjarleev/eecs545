import json
import importlib
import torch
from option.option import get_option
from util.visualizer import Visualizer

from solver import Solver
from data import generate_loader
from model import StudioLightGenerator, RandomLightGenerator, MultiScaleDiscriminator
import numpy as np
import random
import os


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
    if not opt.test_only:
        file_name = os.path.join(opt.ckpt_root, opt.name, 'opt.txt')
        args = vars(opt)
        with open(file_name, 'w+') as opt_file:
            opt_file.write('--------------- Options ---------------\n')
            for k, v in args.items():
                opt_file.write('%s: %s\n' % (str(k), str(v)))
                print('%s: %s' % (str(k), str(v)))
            opt_file.write('----------------- End -----------------\n')

    rand_G = RandomLightGenerator(input_dim=opt.input_dim,
                                  output_dim=opt.output_dim,
                                  noise_dim=opt.noise_dim,
                                  num_downsample=opt.num_downsample,
                                  num_resblock=opt.num_resblock,
                                  ngf=opt.ngf,
                                  padding_mode=opt.paddding_mode_G,
                                  max_channel=opt.max_channel)
    studio_G = None
    studio_D = None
    if not opt.test_only:
        studio_G = StudioLightGenerator(input_dim=opt.input_dim,
                                        output_dim=opt.output_dim,
                                        noise_dim=opt.noise_dim,
                                        num_downsample=opt.num_downsample,
                                        num_resblock=opt.num_resblock,
                                        ngf=opt.ngf,
                                        padding_mode=opt.paddding_mode_G,
                                        max_channel=opt.max_channel)
        studio_D = MultiScaleDiscriminator(input_nc=opt.input_dim,
                                           num_D=opt.num_D,
                                           n_layer=opt.n_lyaer_D,
                                           ndf=opt.ndf,
                                           padding_mode=opt.padding_mode)
    solver = Solver(rand_G, studio_G, studio_D)
    if opt.test_only:
        print('Evaluate {} (loaded from {})'.format(opt.netG, opt.pretrain))
        psnr = solver.evaluate(solver.validation_loader, 'test')
        print("{:.2f}".format(psnr))
    elif opt.infer:
        dataloader = generate_loader(opt, 'infer')
        solver.inference(dataloader, opt.infer_name)
    else:
        solver.fit()

if __name__ == '__main__':
    main()

