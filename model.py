import numpy as np
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

from nnabla.initializer import NormalInitializer


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
