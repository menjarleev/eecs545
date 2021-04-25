import json
from option.option import get_option

from data import *
from torch.utils.data import DataLoader
from model import *
import os
from utils import LossCollector, Visualizer
from utils.transform import *
from functools import partial


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
                                  lc_nc=opt.lc_nc,
                                  num_downsample=opt.num_downsample,
                                  num_resblock=opt.num_resblock,
                                  ngf=opt.ngf,
                                  padding_mode=opt.padding_mode_G,
                                  max_channel=opt.max_channel)
    studio_G = StudioLightGenerator(input_dim=opt.input_dim,
                                    output_dim=opt.output_dim,
                                    lc_nc=opt.lc_nc,
                                    num_downsample=opt.num_downsample,
                                    num_resblock=opt.num_resblock,
                                    ngf=opt.ngf,
                                    padding_mode=opt.padding_mode_G,
                                    max_channel=opt.max_channel)
    lc_G = define_encoder(opt)
    rand_D, lc_D = None, None
    train_dataloader, valid_dataloader = None, None
    if opt.train:
        rand_D = MultiScaleDiscriminator(input_nc=opt.input_dim * 2 + opt.lc_nc,
                                         num_D=opt.num_D,
                                         n_layer=opt.n_layer_D,
                                         ndf=opt.ndf,
                                         padding_mode=opt.padding_mode_D)
        t = [partial(crop, psize=opt.patch_size),
             flip,
             rotate,
             partial(pixel_shift, shift_range=opt.shift_range)]
        transform = Transform(transforms=t)
        train_dataset = eval(f"{opt.dataset}(dataset_root='{opt.dataset_root}', phase='train', transform=transform)")
        train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
        # create dataloader
        if opt.validation:
            valid_dataset = eval(f"{opt.dataset}(dataset_root='{opt.dataset_root}', phase='valid', transform=None,"
                                f"use_ref={opt.use_ref}, num_lighting_infer={opt.num_lighting_infer})")
            valid_dataloader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)

    solver = PhotometricGAN(rand_G, lc_G, studio_G, rand_D, lc_D, gpu_id=opt.gpu_id)
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
                                       lambda_vae=opt.lambda_vae,
                                       lambda_kl=opt.lambda_kl,)
        solver.fit(gpu_id=opt.gpu_id,
                   lr=opt.lr,
                   save_dir=opt.save_dir,
                   max_step=opt.max_step,
                   finetune=opt.finetune,
                   finetune_step=opt.finetune_step,
                   gamma=opt.gamma,
                   decay=opt.decay,
                   loss_collector=loss_collector,
                   visualizer=visualizer,
                   step_label=opt.step_label,
                   train_dataloader=train_dataloader,
                   val_dataloader=valid_dataloader,
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
        test_dataset = eval(f"{opt.dataset}(dataset_root='{opt.dataset_root}', phase='test', transform=None,"
                            f"use_ref={opt.use_ref}, num_lighting_infer={opt.num_lighting_infer})")
        test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size_eval, shuffle=False)
        solver.test(
            gpu_id=opt.gpu_id,
            save_dir=opt.save_dir,
            visualizer=visualizer,
            test_dataloader=test_dataloader,
            save_result=True,
            step_label=opt.step_label,
            test_step=opt.test_step)

    if opt.inference:
        # create dataloader
        # substitute train with infer dataset you like
        test_dataset = eval(f"{opt.dataset}(dataset_root='{opt.dataset_root}', phase='test', transform=None,"
                            f"use_ref={opt.use_ref}, num_lighting_infer={opt.num_lighting_infer})")
        test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size_eval, shuffle=False)
        solver.inference(gpu_id=opt.gpu_id,
                         dataloader=test_dataloader,
                         save_dir=opt.save_dir,
                         num_lighting_infer=opt.num_lighting_infer,
                         label=opt.label_infer,
                         visualizer=visualizer)


if __name__ == '__main__':
    main()
