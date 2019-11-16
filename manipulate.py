import numpy as np
import nnabla as nn
import model
import os

from functools import partial
from helper import denormalize, imwrite, rescale_generated_images
from train import _pad
from nnabla.ext_utils import get_extension_context


def generate(args, Zs, reals, noise_amps, gen_start=0, num_samples=50):
    if args.gpu:
        ctx = get_extension_context('cudnn', device_id='0')
        nn.set_default_context(ctx)

    # generator model
    generator = partial(model.generator, num_layer=args.num_layer,
                        fs=args.fs, min_fs=args.min_fs, kernel=args.kernel,
                        pad=args.pad, test=True)

    images_curr = []
    for scale_num, (Z_opt, noise_amp) in enumerate(zip(Zs, noise_amps)):
        real = reals[scale_num]

        images_prev = images_curr
        images_curr = []

        # inference graph
        x = nn.Variable((1, 3, real.shape[2], real.shape[3]))
        y = nn.Variable((1, 3, real.shape[2], real.shape[3]))
        padded_x = _pad(x, args.kernel, args.num_layer)
        fake = generator(x=padded_x, y=y, scope='g%d' % scale_num)

        for i in range(num_samples):
            if scale_num >= gen_start:
                z_curr = np.random.normal(
                    0.0, 1.0, size=(1, 3, real.shape[2], real.shape[3]))
            else:
                z_curr = Z_opt

            if scale_num == 0:
                I_prev = np.zeros_like(real)
            else:
                I_prev = rescale_generated_images(
                    images_prev[i], 1 / args.scale_factor)
                I_prev = I_prev[:, :, :real.shape[2], :real.shape[3]]

            # generate image
            x.d = noise_amp * z_curr + I_prev
            y.d = I_prev
            fake.forward(clear_buffer=True)

            I_curr = fake.d

            if scale_num == len(reals) - 1:
                path = os.path.join(args.logdir, 'generated_image%d.png' % i)
                image = np.transpose(I_curr[0], [1, 2, 0])
                imwrite(denormalize(image), path)

            images_curr.append(I_curr)
