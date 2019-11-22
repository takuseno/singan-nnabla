import numpy as np
import nnabla as nn
import math
import os

from model import Model
from helper import imread, imwrite, imresize, imrescale, create_real_images
from helper import normalize, denormalize, save_pkl, rescale_generated_images
from nnabla.monitor import Monitor, MonitorSeries, MonitorImage
from nnabla.utils.image_utils import set_backend


def train(args):
    # create real images
    reals = create_real_images(args)
    # save real images
    for i, real in enumerate(reals):
        image_path = os.path.join(args.logdir, 'real_%d.png' % i)
        imwrite(denormalize(np.transpose(real, [0, 2, 3, 1])[0]), image_path)

    # nnabla monitor
    monitor = Monitor(args.logdir)
    # use cv2 backend at MonitorImage
    set_backend('cv2')

    prev_models = []
    Zs = []
    noise_amps = []

    for scale_num in range(args.stop_scale + 1):
        fs = min(args.fs_init * (2 ** (scale_num // 4)), 128)
        min_fs = min(args.min_fs_init * (2 ** (scale_num // 4)), 128)

        model = Model(reals[scale_num], args.num_layer, fs, min_fs,
                      args.kernel, args.pad, args.lam_grad, args.alpha_recon,
                      args.d_lr, args.g_lr, args.beta1, args.gamma,
                      args.lr_milestone, str(scale_num))

        z_curr = train_single_scale(args, scale_num, model, reals,
                                    prev_models, Zs, noise_amps, monitor)

        prev_models.append(model)
        Zs.append(z_curr)
        noise_amps.append(args.noise_amp)

    # save data
    nn.save_parameters(os.path.join(args.logdir, 'models.h5'))
    save_pkl(Zs, os.path.join(args.logdir, 'Zs.pkl'))
    save_pkl(reals, os.path.join(args.logdir, 'reals.pkl'))
    save_pkl(noise_amps, os.path.join(args.logdir, 'noise_amps.pkl'))

    return Zs, reals, noise_amps


def train_single_scale(args, index, model, reals, prev_models, Zs,
                       noise_amps, monitor):
    # prepare log monitors
    monitor_train_d_real = MonitorSeries('train_d_real%d' % index, monitor)
    monitor_train_d_fake = MonitorSeries('train_d_fake%d' % index, monitor)
    monitor_train_g_fake = MonitorSeries('train_g_fake%d' % index, monitor)
    monitor_train_g_rec = MonitorSeries('train_g_rec%d' % index, monitor)
    monitor_image_g = MonitorImage('image_g_%d' % index, monitor, interval=1,
                                   num_images=1, normalize_method=denormalize)

    real = reals[index]
    ch, w, h = real.shape[1], real.shape[2], real.shape[3]

    # training loop
    for epoch in range(args.niter):
        d_real_error_history = []
        d_fake_error_history = []
        g_fake_error_history = []
        g_rec_error_history = []

        if index == 0:
            z_opt = np.random.normal(0.0, 1.0, size=(1, 1, w, h))
            noise_ = np.random.normal(0.0, 1.0, size=(1, 1, w, h))
        else:
            z_opt = np.zeros((1, ch, w, h))
            noise_ = np.random.normal(0.0, 1.0, size=(1, ch, w, h))

        # discriminator training loop
        for d_step in range(args.d_steps):
            # previous outputs
            if d_step == 0 and epoch == 0:
                if index == 0:
                    prev = np.zeros_like(noise_)
                    z_prev = np.zeros_like(z_opt)
                    args.noise_amp = 1
                else:
                    prev = _draw_concat(args, index, prev_models, Zs, reals,
                                        noise_amps, 'rand')
                    z_prev = _draw_concat(args, index, prev_models, Zs,
                                          reals, noise_amps, 'rec')
                    rmse = np.sqrt(np.mean((real - z_prev) ** 2))
                    args.noise_amp = args.noise_amp_init * rmse
            else:
                prev = _draw_concat(args, index, prev_models, Zs, reals,
                                    noise_amps, 'rand')

            # input noise
            if index == 0:
                noise = noise_
            else:
                noise = args.noise_amp * noise_ + prev

            fake_error, real_error = model.update_discriminator(epoch, noise,
                                                                prev)
            # accumulate errors for logging
            d_real_error_history.append(real_error)
            d_fake_error_history.append(fake_error)

        # generator training loop
        for g_step in range(args.g_steps):
            fake_error, rec_error = model.update_generator(
                epoch, noise, prev, args.noise_amp * z_opt + z_prev, z_prev)
            # accumulate errors for logging
            g_fake_error_history.append(fake_error)
            g_rec_error_history.append(rec_error)

        # save errors
        monitor_train_d_real.add(epoch, np.mean(d_real_error_history))
        monitor_train_d_fake.add(epoch, np.mean(d_fake_error_history))
        monitor_train_g_fake.add(epoch, np.mean(g_fake_error_history))
        monitor_train_g_rec.add(epoch, np.mean(g_rec_error_history))

        # save generated image
        monitor_image_g.add(epoch, model.generate(noise, prev))

    return z_opt


def _draw_concat(args, index, prev_models, Zs, reals, noise_amps, mode):
    G_z = np.zeros_like(reals[0])
    if index > 0:
        pad_noise = int(((args.kernel - 1) * args.num_layer) / 2)
        for i in range(index):
            Z_opt = Zs[i]
            real_curr = reals[i]
            real_next = reals[i + 1]
            noise_amp = noise_amps[i]
            G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]

            if mode == 'rand':
                if i == 0:
                    z_shape = (1, 1) + real_curr.shape[2:]
                else:
                    z_shape = (1,) + real_curr.shape[1:]
                z = np.random.normal(0.0, 1.0, size=z_shape)
                Z_in = noise_amp * z + G_z
            elif mode == 'rec':
                Z_in = noise_amp * Z_opt + G_z
            else:
                raise Exception

            # generate image with previous output and noise
            G_z = prev_models[i].generate(Z_in, G_z)
            G_z = rescale_generated_images(G_z, 1 / args.scale_factor)
            G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
    return G_z
