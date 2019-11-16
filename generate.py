import nnabla as nn

from manipulate import generate
from helper import load_pkl
from args import get_args


def main(args):
    reals = load_pkl(args.load_reals)
    Zs = load_pkl(args.load_Zs)
    noise_amps = load_pkl(args.load_noise_amps)
    nn.load_parameters(args.load)

    generate(args, Zs, reals, noise_amps)


if __name__ == '__main__':
    args = get_args()
    main(args)
