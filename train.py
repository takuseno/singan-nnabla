import numpy as np
import nnabla as nn
import nnabla.solvers as S
import nnabla.functions as F
import math
import model

from functools import partial
from helper import imread, imresize, imrescale, create_reals_pyramid
from nnabla.monitor import Monitor, MonitorSeries, MonitorImage


def _calc_gradient_penalty(real, fake, discriminator, scope):
    alpha = F.rand(shape=real.shape)
    interpolates = alpha * real + (1.0 - alpha) * fake

    disc_interpolates = discriminator(x=interpolates, scope=scope)

    with nn.parameter_scope(scope):
        params = nn.get_parameters()

    grads = nn.grad([disc_interpolates], params.values())

    norms = [((F.sum(g ** 2.0) ** 0.5) - 1.0) ** 2.0 for g in grads]
    penalty = sum([F.mean(norm) for norm in norms])

    return penalty


def _create_real_images(args):
    real_raw = imread(args.image_path)
    edge = max([real_raw.shape[0], real_raw.shape[1]])

    args.num_scales = int((math.log(args.min_size / (real_raw.shape[1]),
                                    args.scale_factor_init))) + 1
    print('num_scales', args.num_scales)

    scale2stop = int(math.log(min([args.max_size, edge]) / edge,
                              args.scale_factor_init))
    print('scale2stop', scale2stop)

    args.stop_scale = args.num_scales - scale2stop

    args.scale1 = min(args.max_size / edge, 1)
    print('scale1', args.scale1)

    real = imrescale(real_raw, args.scale1)
    print('real image shape', real.shape)

    args.scale_factor = (args.min_size / real.shape[0]) ** (1 / args.stop_scale)

    scale = min(args.max_size / edge, 1)
    real = imrescale(real_raw, scale)
    print('image shape:', real.shape)

    reals = create_reals_pyramid(real, args.stop_scale, args.scale_factor)
    for i in range(len(reals)):
        print('%d: real shape:' % i, reals[i].shape)

    return reals


def train(args):
    # generator model
    generator = partial(model.generator, num_layer=args.num_layer,
                        fs=args.fs, min_fs=args.min_fs, kernel=args.kernel,
                        pad=args.pad)

    # discriminator model
    discriminator = partial(model.discriminator, num_layer=args.num_layer,
                            fs=args.fs, min_fs=args.min_fs, kernel=args.kernel,
                            pad=args.pad)

    # create real images
    reals = _create_real_images(args)

    # nnabla monitor
    monitor = Monitor(args.logdir)

    Zs = []
    in_s = 0
    noise_amps = []
    scale_num = 0

    while scale_num < args.stop_scale + 1:
        fs = min(args.fs_init * 2 ** (scale_num // 4), 128)
        min_fs = min(args.min_fs_init * 2 ** (scale_num // 4), 128)

        z_curr, in_s = train_single_scale(args, scale_num, generator,
                                          discriminator, reals, Zs, in_s,
                                          noise_amps, monitor)

        Zs.append(z_curr)
        noise_amps.append(args.noise_amp)

        scale_num += 1



def train_single_scale(args, index, generator, discriminator, reals, Zs, in_s,
                       noise_amps, monitor):
    # prepare log monitors
    monitor_train_d_real = MonitorSeries('train_d_real%d' % index, monitor)
    monitor_train_d_fake = MonitorSeries('train_d_fake%d' % index, monitor)
    monitor_train_g_fake = MonitorSeries('train_g_fake%d' % index, monitor)
    monitor_train_g_rec = MonitorSeries('train_g_rec%d' % index, monitor)
    normalize_method = lambda x: np.array(255 * (x / 2 + 0.5), dtype=np.uint8)
    monitor_image_g = MonitorImage('image_g_%d' % index, monitor, interval=1,
                                   num_images=1,
                                   normalize_method=normalize_method)

    real = np.array(reals[index])

    ch, w, h = real.shape[1], real.shape[2], real.shape[3]
    pad = int((args.kernel - 1) * args.num_layer / 2)

    # input noise
    fixed_noise = np.random.normal(0.0, 1.0, size=(1, ch, w, h))
    z_opt = np.zeros_like(fixed_noise)

    # inputs
    x = nn.Variable((1, ch, w, h))
    y = nn.Variable((1, ch, w, h))
    rec_x = nn.Variable((1, ch, w, h))
    rec_y = nn.Variable((1, ch, w, h))
    y_real = nn.Variable.from_numpy_array(real)
    y_real.persistent = True

    # padding
    padded_x = F.pad(x, (pad, pad, pad, pad))
    padded_rec_x = F.pad(rec_x, (pad, pad, pad, pad))

    # parameter scopes of discriminator and generator
    d_scope = 'd%d' % index
    g_scope = 'g%d' % index

    # training graph
    p_real = discriminator(x=y_real, scope=d_scope)
    fake = generator(x=padded_x, y=y, scope=g_scope)
    p_fake = discriminator(x=fake, scope=d_scope)

    # gradient penalty for discriminator
    grad_penalty = _calc_gradient_penalty(y_real, fake, discriminator, d_scope)

    # discriminator loss
    d_real_error = -F.mean(p_real)
    d_fake_error = F.mean(p_fake)
    d_error = d_real_error + d_fake_error + args.lam_grad * grad_penalty

    # generator loss
    rec = generator(x=padded_rec_x, y=rec_y, scope=g_scope)
    rec_loss = F.mean(F.squared_error(rec, y_real))
    g_fake_error = -F.mean(p_fake)
    g_error = g_fake_error + args.alpha_recon * rec_loss

    # prepare training parameters
    with nn.parameter_scope(d_scope):
        d_params = nn.get_parameters()
    with nn.parameter_scope(g_scope):
        g_params = nn.get_parameters()

    # create solvers
    d_solver = S.Adam(args.d_lr)
    d_solver.set_parameters(d_params)
    g_solver = S.Adam(args.g_lr)
    g_solver.set_parameters(g_params)

    # training loop
    for epoch in range(args.niter):
        sum_d_real_error = 0.0
        sum_d_fake_error = 0.0
        sum_g_fake_error = 0.0
        sum_g_rec_error = 0.0

        if index == 0:
            z_opt = np.random.normal(0.0, 1.0, size=(1, 3, w, h))
            noise_ = np.random.normal(0.0, 1.0, size=(1, 3, w, h))

        # discriminator training loop
        for d_step in range(args.d_steps):
            # previous outputs
            if d_step == 0 and epoch == 0:
                if index == 0:
                    prev = np.zeros_like(noise_)
                    in_s = prev
                    z_prev = np.zeros_like(z_opt)
                    args.noise_amp = 1
                else:
                    prev = _draw_concat(args, index, generator, Zs, reals,
                                        noise_amps, in_s, 'rand')
                    z_prev = _draw_concat(args, index, generator, Zs, reals,
                                          noise_amps, in_s, 'rec')
                    rmse = np.sqrt(np.mean((real - z_prev) ** 2))
                    args.noise_amp = args.noise_amp_init * rmse
            else:
                prev = _draw_concat(args, index, generator, Zs, reals,
                                    noise_amps, in_s, 'rec')

            # input noise
            if index == 0:
                noise = noise_
            else:
                noise = args.noise_amp * noise_ + prev

            x.d = noise
            y.d = prev
            d_error.forward()

            # accumulate errors for logging
            sum_d_real_error += d_real_error.d
            sum_d_fake_error += d_fake_error.d

            d_solver.zero_grad()
            d_error.backward(clear_buffer=True)
            d_solver.update()

        # generator training loop
        for g_step in range(args.g_steps):
            x.d = noise
            y.d = prev
            rec_x.d = args.noise_amp * z_opt + z_prev
            rec_y.d = z_prev
            g_error.forward()

            # accumulate errors for logging
            sum_g_fake_error += g_fake_error.d
            sum_g_rec_error += rec_loss.d

            g_solver.zero_grad()
            g_error.backward(clear_buffer=True)
            g_solver.update()

        # calculate mean errors
        mean_d_real_error = sum_d_real_error / args.niter / args.d_steps
        mean_d_fake_error = sum_d_fake_error / args.niter / args.d_steps
        mean_g_fake_error = sum_g_fake_error / args.niter / args.g_steps
        mean_g_rec_error = sum_g_rec_error / args.niter / args.g_steps

        # save errors
        monitor_train_d_real.add(epoch, mean_d_real_error)
        monitor_train_d_fake.add(epoch, mean_d_fake_error)
        monitor_train_g_fake.add(epoch, mean_g_fake_error)
        monitor_train_g_rec.add(epoch, mean_g_rec_error)

        # save generated image
        fake.forward(clear_buffer=True)
        monitor_image_g.add(epoch, fake.d)

    return z_opt, in_s


def _draw_concat(args, index, generator, Zs, reals, noise_amps, in_s, mode):
    G_z = in_s
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
                    z_shape = (1, Z_opt.shape[2] - 2 * pad_noise, \
                                  Z_opt.shape[3] - 2 * pad_noise)
                else:
                    z_shape = (3, Z_opt.shape[2] - 2 * pad_noise, \
                                  Z_opt.shape[3] - 2 * pad_noise)
                z = np.random.normal(0.0, 1.0, size=z_shape)
                Z_in = noise_amp * z + G_z
            elif mode == 'rec':
                Z_in = noise_amp * Z_opt + G_z
            else:
                raise Exception
            x = nn.Variable.from_numpy_array(z_in)
            y = nn.Variable.from_numpy_array(G_z)
            fake = generator(x=x, y=y, scope='g%d' % i)
            fake.forward(clear_buffer=True)
            G_z = fake.d
            G_z = imscale(G_z, 1 / args.scale_factor)
            G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
    return G_z
