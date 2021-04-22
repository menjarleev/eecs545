import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)

    # models
    parser.add_argument('--netG', type=str, default='Cycle')
    parser.add_argument('--netD', type=str, default='Multiscale')
    parser.add_argument('--actv_G', type=str, default='ReLU')
    parser.add_argument('--num_downsample', type=int, default=3)
    parser.add_argument('--num_resblock', type=int, default=8)
    parser.add_argument('--padding_mode_G', type=str, default='reflect')
    parser.add_argument('--actv_D', type=str, default='LeakyReLU')
    parser.add_argument('--padding_mode_D', type=str, default='zeros')
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--n_layer_D', type=int, default=3)
    parser.add_argument('--num_D', type=int, default=2)
    parser.add_argument('--gan_mode', type=str, default='hinge', help='[ls|origin|hinge]')
    parser.add_argument('--norm_D', type=str, default='instance')
    parser.add_argument('--use_vgg', action='store_true')
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--max_channel', type=int, default=256)


    # dataset
    parser.add_argument('--dataset_root', type=str, default='./')

    # training setups
    parser.add_argument('--optim_name', type=str, default='Adam',
                        choices=['Adam', 'SGD'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--decay', type=str, default='150-250-350')
    parser.add_argument('--gamma', type=int, default=0.5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--max_step', type=int, default=100000)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=0)

    # misc
    parser.add_argument('--save_result', action='store_true')
    parser.add_argument('--ckpt_root', type=str, default='./ckpt')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--lambda_feat', type=float, default=10.0)
    parser.add_argument('--lambda_L1', type=float, default=3.0)
    parser.add_argument('--lambda_vgg', type=float, default=10.0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--name', type=str, default='photometricGAN')
    # parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--print_mem', type=bool, default=True)
    parser.add_argument('--step_label', type=str, default='latest')
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--loss_terms', type=str, default=['L1', 'GAN', 'vgg'], nargs='+')
    parser.add_argument('--gpu_id', type=int, default=0)

    # test
    parser.add_argument('--train', action='store_true', dest='train')
    parser.add_argument('--validation', action='store_true', dest='validation')
    parser.add_argument('--validation_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--test', action='store_true', dest='test')
    parser.add_argument('--test_step', type=int, default=50)

    # photometrics specific options
    parser.add_argument('--num_lighting', type=int, default=9)
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--bottleneck_z', type=int, default=256)
    parser.add_argument('--num_conv_z', type=int, default=2)
    parser.add_argument('--num_linear_z', type=int, default=4)

    # inference
    parser.add_argument('--inference', action='store_true', dest='inference')
    parser.add_argument('--num_lighting_infer', type=int, default=9)
    parser.add_argument('--label_infer', type=str, default='best', choices=['best', 'latest'])
    parser.add_argument('--latent_size', type=int, nargs='+', default=(16, 8, 8))

    # agumentation
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--shift_range', type=int, nargs='+', default=(-20, 20))


    return parser.parse_args()


def get_option():
    opt = parse_args()
    if not os.path.exists(f"{opt.ckpt_root}"):
        os.makedirs(f"{opt.ckpt_root}")   # Added to avoid path not exist error on Win10

    n = len([x for x in os.listdir(opt.ckpt_root) if x.startswith(opt.name)])
    save_dir = os.path.join(opt.ckpt_root, f'{opt.name}_{n + 1}')
    if opt.continue_train or opt.model_dir is not None:
        save_dir = opt.model_dir
    else:
        os.makedirs(save_dir, exist_ok=False)
    setattr(opt, 'save_dir', save_dir)
    opt.loss_terms = [loss.lower() for loss in opt.loss_terms]
    return opt

