import nnabla as nn

from args import get_args
from train import train
from manipulate import generate
from nnabla.ext_utils import get_extension_context


if __name__ == '__main__':
    args = get_args()

    if args.gpu:
        ctx = get_extension_context('cudnn', device_id='0')
        nn.set_default_context(ctx)

    Zs, reals, noise_amps = train(args)
    generate(args, Zs, reals, noise_amps)
