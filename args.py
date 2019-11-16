import argparse
import os

from datetime import datetime


def get_args():
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    parser = argparse.ArgumentParser('NNabla implementation of SinGAN.')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--g-lr', type=float, default=5e-4)
    parser.add_argument('--d-lr', type=float, default=5e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--lr-milestone', type=int, default=1600)
    parser.add_argument('--fs', type=int, default=32)
    parser.add_argument('--min-fs', type=int, default=32)
    parser.add_argument('--kernel', type=int, default=3)
    parser.add_argument('--num-layer', type=int, default=5)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--pad', type=int, default=0)
    parser.add_argument('--scale-factor', type=float, default=0.75)
    parser.add_argument('--min-size', type=int, default=25)
    parser.add_argument('--max-size', type=int, default=256)
    parser.add_argument('--noise-amp', type=float, default=0.1)
    parser.add_argument('--d-steps', type=int, default=3)
    parser.add_argument('--g-steps', type=int, default=3)
    parser.add_argument('--niter', type=int, default=2000)
    parser.add_argument('--lam-grad', type=float, default=0.1)
    parser.add_argument('--alpha-recon', type=float, default=10.0)
    parser.add_argument('--image-path', type=str)
    parser.add_argument('--load', type=str)
    parser.add_argument('--load-reals', type=str)
    parser.add_argument('--load-noise-amps', type=str)
    parser.add_argument('--load-Zs', type=str)
    parser.add_argument('--gen-start', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='experiment')
    args = parser.parse_args()

    args.logdir = os.path.join('logs', args.logdir + '_' + date)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # additional setup
    args.niter_init = args.niter
    args.noise_amp_init = args.noise_amp
    args.fs_init = args.fs
    args.min_fs_init = args.min_fs
    args.scale_factor_init = args.scale_factor

    return args
