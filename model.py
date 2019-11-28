import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

from functools import partial
from nnabla.initializer import NormalInitializer
from nnabla.utils.learning_rate_scheduler import StepScheduler


def conv_block(x, out_ch, kernel, stride, pad, test, scope):
    with nn.parameter_scope(scope):
        h = PF.convolution(x, out_ch, (kernel, kernel), stride=(stride, stride),
                           pad=(pad, pad), w_init=NormalInitializer(0.02),
                           name='conv')
        h = PF.batch_normalization(h, batch_stat=not test, name='bn')
        h = F.leaky_relu(h, alpha=0.2)
    return h


def middle_blocks(x, num_layer, fs, min_fs, kernel, pad, test):
    h = x
    for i in range(num_layer - 2):
        N = int(fs / (2 ** (i + 1)))
        ch = max(N, min_fs)
        h = conv_block(h, ch, kernel, 1, pad, test, 'block%d' % (i + 1))
    return h


def generator(x, y, num_layer, fs, min_fs, kernel, pad, scope, test=False):
    with nn.parameter_scope(scope):
        h = conv_block(x, fs, kernel, 1, pad, test, 'head')
        h = middle_blocks(h, num_layer, fs, min_fs, kernel, pad, test)
        h = PF.convolution(h, x.shape[1], (kernel, kernel),
                           stride=(1, 1), pad=(pad, pad),
                           w_init=NormalInitializer(0.02), name='tail')
        h = F.tanh(h)
    ind = int((y.shape[2] - h.shape[2]) / 2)
    y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
    return y + h


def discriminator(x, num_layer, fs, min_fs, kernel, pad, scope, test=False):
    with nn.parameter_scope(scope):
        h = conv_block(x, fs, kernel, 1, pad, test, 'head')
        h = middle_blocks(h, num_layer, fs, min_fs, kernel, pad, test)
        h = PF.convolution(h, 1, (kernel, kernel),
                           stride=(1, 1), pad=(pad, pad),
                           w_init=NormalInitializer(0.02), name='tail')
    return h


def _pad(x, kernel, num_layer):
    pad = int((kernel - 1) * num_layer / 2)
    return F.pad(x, (pad, pad, pad, pad))


# one-centered gradient penalty
def _calc_gradient_penalty(real, fake, discriminator):
    alpha = F.rand(shape=(1, 1, 1, 1))
    interpolates = alpha * real + (1.0 - alpha) * fake
    interpolates.need_grad = True

    disc_interpolates = discriminator(x=interpolates)

    grads = nn.grad([disc_interpolates], [interpolates])
    norms = [F.sum(g ** 2.0, axis=1) ** 0.5 for g in grads]
    return sum([F.mean((norm - 1.0) ** 2.0) for norm in norms])


class Model:
    def __init__(self, real, num_layer, fs, min_fs, kernel, pad, lam_grad,
                 alpha_recon, d_lr, g_lr, beta1, gamma, lr_milestone, scope,
                 test=False):
        self.real = real
        self.num_layer = num_layer
        self.fs = fs
        self.min_fs = min_fs
        self.kernel = kernel
        self.pad = pad
        self.lam_grad = lam_grad
        self.alpha_recon = alpha_recon
        self.d_lr = d_lr
        self.g_lr = g_lr
        self.beta1 = beta1
        self.gamma = gamma
        self.lr_milestone = lr_milestone
        self.scope = scope
        self.test = test

        self._build()
        self._setup_solvers()


    # for generating larger images than the ones in training time
    def get_generator_func(self, image_shape, channel_first=True):
        if not channel_first:
            image_shape = (image_shape[2], image_shape[0], image_shape[1])
        generator_fn, _ = self._network_funcs()

        # build inference graph
        x_ = nn.Variable((1,) + image_shape)
        y_ = nn.Variable((1,) + image_shape)
        fake = generator_fn(x=x, y=y)

        def func(x, y):
            x_.d = x
            y_.d = y
            fake.forward(clear_buffer=True)
            return fake.d.copy()

        return func


    def _network_funcs(self):
        # generator model
        generator_fn = partial(generator, num_layer=self.num_layer, fs=self.fs,
                               min_fs=self.min_fs, kernel=self.kernel,
                               pad=self.pad, scope='%s/generator' % self.scope,
                               test=self.test)
        # discriminator model
        discriminator_fn = partial(discriminator, num_layer=self.num_layer,
                                   fs=self.fs, min_fs=self.min_fs,
                                   kernel=self.kernel, pad=self.pad,
                                   scope='%s/discriminator' % self.scope,
                                   test=self.test)
        return generator_fn, discriminator_fn


    def _build(self):
        generator_fn, discriminator_fn = self._network_funcs()

        # real shape
        ch, w, h = self.real.shape[1:]

        # inputs
        self.x = nn.Variable((1, ch, w, h))
        self.y = nn.Variable((1, ch, w, h))
        self.rec_x = nn.Variable((1, ch, w, h))
        self.rec_y = nn.Variable((1, ch, w, h))
        y_real = nn.Variable.from_numpy_array(self.real)
        y_real.persistent = True

        # padding inputs
        padded_x = _pad(self.x, self.kernel, self.num_layer)
        padded_rec_x = _pad(self.rec_x, self.kernel, self.num_layer)

        # generate fake image
        self.fake = generator_fn(x=padded_x, y=self.y)
        fake_without_grads = F.identity(self.fake)
        fake_without_grads.need_grad = False
        rec = generator_fn(x=padded_rec_x, y=self.rec_y)

        # discriminate images
        p_real = discriminator_fn(x=y_real)
        p_fake = discriminator_fn(x=self.fake)
        p_fake_without_grads = discriminator_fn(x=fake_without_grads)

        # gradient penalty for discriminator
        grad_penalty = _calc_gradient_penalty(y_real, fake_without_grads,
                                              discriminator_fn)

        # discriminator loss
        self.d_real_error = -F.mean(p_real)
        self.d_fake_error = F.mean(p_fake_without_grads)
        self.d_error = self.d_real_error + self.d_fake_error \
                                         + self.lam_grad * grad_penalty

        # generator loss
        self.rec_error = F.mean(F.squared_error(rec, y_real))
        self.g_fake_error = -F.mean(p_fake)
        self.g_error = self.g_fake_error + self.alpha_recon * self.rec_error


    def _setup_solvers(self):
        # prepare training parameters
        with nn.parameter_scope('%s/discriminator' % self.scope):
            d_params = nn.get_parameters()
        with nn.parameter_scope('%s/generator' % self.scope):
            g_params = nn.get_parameters()

        # create solver for discriminator
        self.d_lr_scheduler = StepScheduler(self.d_lr, self.gamma,
                                            [self.lr_milestone])
        self.d_solver = S.Adam(self.d_lr, beta1=self.beta1, beta2=0.999)
        self.d_solver.set_parameters(d_params)

        # create solver for generator
        self.g_lr_scheduler = StepScheduler(self.g_lr, self.gamma,
                                            [self.lr_milestone])
        self.g_solver = S.Adam(self.g_lr, beta1=self.beta1, beta2=0.999)
        self.g_solver.set_parameters(g_params)


    def generate(self, x, y):
        self.x.d = x
        self.y.d = y
        self.fake.forward(clear_buffer=True)
        return self.fake.d


    def update_g(self, epoch, x, y, rec_x, rec_y):
        self.x.d = x
        self.y.d = y
        self.rec_x.d = rec_x
        self.rec_y.d = rec_y
        self.g_error.forward()
        fake_error = self.g_fake_error.d.copy()
        rec_error = self.rec_error.d.copy()
        self.g_solver.zero_grad()
        self.g_error.backward(clear_buffer=True)
        lr = self.g_lr_scheduler.get_learning_rate(epoch)
        self.g_solver.set_learning_rate(lr)
        self.g_solver.update()
        return fake_error, rec_error


    def update_d(self, epoch, x, y):
        self.x.d = x
        self.y.d = y
        self.d_error.forward()
        real_error = self.d_real_error.d.copy()
        fake_error = self.d_fake_error.d.copy()
        self.d_solver.zero_grad()
        self.d_error.backward(clear_buffer=True)
        lr = self.d_lr_scheduler.get_learning_rate(epoch)
        self.d_solver.set_learning_rate(lr)
        self.d_solver.update()
        return fake_error, real_error
