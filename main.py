from args import get_args
from train import train
from manipulate import generate


if __name__ == '__main__':
    args = get_args()
    Zs, reals, noise_amps = train(args)
    generate(args, Zs, reals, noise_amps)
