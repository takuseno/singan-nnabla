import numpy as np
import os

from model import Model
from helper import denormalize, imwrite, rescale_generated_images


def generate(args, Zs, reals, noise_amps, gen_start=0, num_samples=50):
    images_curr = []
    for scale_num, (Z_opt, noise_amp) in enumerate(zip(Zs, noise_amps)):
        real = reals[scale_num]
        ch, w, h = real.shape[1:]

        fs = min(args.fs_init * (2 ** (scale_num // 4)), 128)
        min_fs = min(args.min_fs_init * (2 ** (scale_num // 4)), 128)
        model = Model(real, args.num_layer, fs, min_fs, args.kernel, args.pad,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, str(scale_num),
                      test=True)

        images_prev = images_curr
        images_curr = []

        for i in range(num_samples):
            if scale_num >= gen_start:
                if scale_num == 0:
                    z_curr = np.random.normal(0.0, 1.0, size=(1, 1, w, h))
                else:
                    z_curr = np.random.normal(0.0, 1.0, size=(1, ch, w, h))
            else:
                z_curr = Z_opt

            if scale_num == 0:
                I_prev = np.zeros_like(real)
            else:
                I_prev = rescale_generated_images(images_prev[i],
                                                  1 / args.scale_factor)
                I_prev = I_prev[:, :, :real.shape[2], :real.shape[3]]

            # generate image
            I_curr = model.generate(noise_amp * z_curr + I_prev, I_prev)

            if scale_num == len(reals) - 1:
                path = os.path.join(args.logdir, 'generated_image%d.png' % i)
                image = np.transpose(I_curr[0], [1, 2, 0])
                imwrite(denormalize(image), path)

            images_curr.append(I_curr)
