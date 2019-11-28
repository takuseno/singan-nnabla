import nnabla as nn

from manipulate import generate
from train import create_real_images
from helper import load_pkl
from args import get_args
from nnabla.ext_utils import get_extension_context


if __name__ == '__main__':
    args = get_args(test=True)

    if args.gpu:
        ctx = get_extension_context('cudnn', device_id='0')
        nn.set_default_context(ctx)

    reals = load_pkl(args.load_reals)
    Zs = load_pkl(args.load_Zs)
    noise_amps = load_pkl(args.load_noise_amps)
    nn.load_parameters(args.load)

    # for setting scale_factor
    create_real_images(args)

    generate(args, Zs, reals, noise_amps, gen_start=args.gen_start)
