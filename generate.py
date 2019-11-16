import nnabla as nn

from manipulate import generate
from train import _create_real_images
from helper import load_pkl
from args import get_args


def main(args):
    reals = load_pkl(args.load_reals)
    Zs = load_pkl(args.load_Zs)
    noise_amps = load_pkl(args.load_noise_amps)
    nn.load_parameters(args.load)

    # for setting scale_factor
    _create_real_images(args)

    generate(args, Zs, reals, noise_amps, gen_start=args.gen_start)


if __name__ == '__main__':
    args = get_args()
    main(args)
