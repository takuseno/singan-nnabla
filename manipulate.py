import numpy as np
import os
import imageio

from model import Model
from helper import denormalize, imwrite, rescale_generated_images
from helper import image_to_batch, batch_to_image


def _get_model(args, scale_num, real):
    fs = min(args.fs_init * (2 ** (scale_num // 4)), 128)
    min_fs = min(args.min_fs_init * (2 ** (scale_num // 4)), 128)
    model = Model(real=real, num_layer=args.num_layer, fs=fs,
                  min_fs=min_fs, kernel=args.kernel, pad=args.pad,
                  lam_grad=0.0, alpha_recon=0.0, d_lr=0.0, g_lr=0.0,
                  beta1=0.0, gamma=0.0, lr_milestone=0,
                  scope=str(scale_num), test=True)
    return model


def generate_animation(args, Zs, reals, noise_amps, frame=100, gen_start=2,
                       alpha=0.1, beta=0.9, fps=10):
    curr_images = []
    for scale_num, (Z_opt, noise_amp) in enumerate(zip(Zs, noise_amps)):
        real = reals[scale_num]
        ch, w, h = real.shape[1:]

        model = _get_model(args, scale_num, real)

        prev_images = curr_images
        curr_images = []

        if scale_num == 0:
            z_rand = np.random.normal(0.0, 1.0, size=(1, 1, w, h))
        else:
            noise = np.random.normal(0.0, 1.0, size=(1, ch, w, h))
        z_prev1 = 0.95 * Z_opt + 0.05 * noise
        z_prev2 = Z_opt

        for i in range(frame):
            if scale_num == 0:
                noise = np.random.normal(0.0, 1.0, size=(1, 1, w, h))
            else:
                noise = np.random.normal(0.0, 1.0, size=(1, ch, w, h))
            diff_curr = beta * (z_prev1 - z_prev2) + (1 - beta) * noise
            z_curr = alpha * Z_opt + (1 - alpha) * (z_prev1 + diff_curr)
            z_prev2 = z_prev1
            z_prev1 = z_curr
            if scale_num < gen_start:
                z_curr = Z_opt

            if scale_num == 0:
                prev_image = np.zeros_like(real)
            else:
                prev_image = rescale_generated_images(prev_images[i],
                                                      1 / args.scale_factor)
                prev_image = prev_image[:, :, :real.shape[2], :real.shape[3]]

            curr_image = model.generate(
                noise_amp * z_curr + prev_image, prev_image)

            curr_images.append(curr_image)

    images = list(map(batch_to_image, curr_images))

    path = os.path.join(args.logdir, 'animation.gif')
    imageio.mimsave(path, images, fps=fps)

    return images


def generate(args, Zs, reals, noise_amps, seed_image=None, gen_start=0,
             num_samples=50):
    if seed_image is None:
        seed_image = np.zeros_like(reals[0])

    curr_images = []
    for scale_num, (Z_opt, noise_amp) in enumerate(zip(Zs, noise_amps)):
        real = reals[scale_num]
        ch, w, h = real.shape[1:]

        model = _get_model(args, scale_num, real)

        prev_images = curr_images
        curr_images = []

        for i in range(num_samples):
            if scale_num >= gen_start:
                if scale_num == 0:
                    z_curr = np.random.normal(0.0, 1.0, size=(1, 1, w, h))
                else:
                    z_curr = np.random.normal(0.0, 1.0, size=(1, ch, w, h))
            else:
                z_curr = Z_opt

            if len(prev_images) == 0:
                prev_image = seed_image
            else:
                prev_image = rescale_generated_images(prev_images[i],
                                                      1 / args.scale_factor)
                prev_image = prev_image[:, :, :real.shape[2], :real.shape[3]]

            # generate image
            curr_image = model.generate(
                noise_amp * z_curr + prev_image, prev_image)

            curr_images.append(curr_image)

    images = list(map(batch_to_image, curr_images))

    for i, image in enumerate(images):
        path = os.path.join(args.logdir, 'generated_image%d.png' % i)
        imwrite(image, path)

    return images


def generate_harmonization(args, Zs, reals, noise_amps, ref, mask, gen_start=2):
    image = generate(args, Zs[gen_start:], reals[gen_start:],
                     noise_amps[gen_start:], seed_image=seed_image)[0]

    composite_image = (1.0 - mask) * real + mask * image_to_batch(image)
    final_image = batch_to_image(composite_image)

    path = os.path.join(args.logdir, 'harmonization_image.png')
    imwrite(final_image, path)

    return final_image
