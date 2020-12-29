from utils import *
from ops import *

import tensorflow as tf


class G_net(object):
    def __init__(self, x_init):
        x_init = ConvBlock(x_init, 64)
        x_init = ConvBlock(x_init, 64)
        x_init = Down_Conv(x_init, 128)

        x_init = ConvBlock(x_init, 128)
        x_init = DSConv(x_init, 128)
        x_init = Down_Conv(x_init, 256)
        x_init = ConvBlock(x_init, 256)

        x_init = InvertedRes_block(x_init, 2, 256, 1, 'r1')
        x_init = InvertedRes_block(x_init, 2, 256, 1, 'r2')
        # x_init = InvertedRes_block(x_init, 2, 256, 1, 'r3')
        # x_init = InvertedRes_block(x_init, 2, 256, 1, 'r4')
        # x_init = InvertedRes_block(x_init, 2, 256, 1, 'r5')
        # x_init = InvertedRes_block(x_init, 2, 256, 1, 'r6')
        # x_init = InvertedRes_block(x_init, 2, 256, 1, 'r7')
        # x_init = InvertedRes_block(x_init, 2, 256, 1, 'r8')

        x_init = ConvBlock(x_init, 256)
        x_init = Up_Conv(x_init, 128)
        x_init = DSConv(x_init, 128)
        x_init = ConvBlock(x_init, 128)

        x_init = Up_Conv(x_init, 128)
        x_init = ConvBlock(x_init, 64)
        x_init = ConvBlock(x_init, 64)

        output = Conv2D(x_init, 3, 1)

        self.fake = tf.tanh(output)