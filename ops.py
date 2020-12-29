import tensorflow as tf
import tensorflow.contrib as tf_contrib
from vgg19 import Vgg19

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None

##################################################################################
# Layer
##################################################################################

def ConvBlock(inputs, filters, kernel_size=3, strides=1, padding='VALID', Use_bias = None):
    x = Conv2D(inputs, filters, kernel_size, strides,padding=padding, Use_bias = Use_bias)
    x = instance_norm(x,scope=None)
    return lrelu(x)


def DSConv(inputs, filters, kernel_size=3, strides=1, padding='VALID', Use_bias = None):
    if kernel_size==3 and strides==1:
        inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
    if strides == 2:
        inputs = tf.pad(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="REFLECT")

    return tf.contrib.layers.separable_conv2d(
        inputs,
        num_outputs=filters,
        kernel_size=kernel_size,
        depth_multiplier=1,
        stride=strides,
        biases_initializer=Use_bias,
        normalizer_fn=tf.contrib.layers.instance_norm,
        activation_fn=lrelu,
        padding=padding)


def Down_Conv(inputs, filters, kernel_size=3, strides=1):

    x = DSConv(filters=filters, kernel_size=kernel_size, strides=2*strides, inputs=inputs)
    new_H, new_W =  tf.shape(inputs)[1] // 2, tf.shape(inputs)[2] // 2
    inputs = tf.image.resize_images(inputs, [new_H, new_W])
    x = x + DSConv(filters=filters, kernel_size=kernel_size, strides= strides, inputs=inputs)

    return x

def Up_Conv(inputs, filters = 256, kernel_size=3):

    new_H, new_W =  tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2
    inputs = tf.image.resize_images(inputs, [new_H, new_W])

    return DSConv(filters=filters, kernel_size=kernel_size, inputs=inputs)


def InvertedRes_block(input, expansion_ratio, output_dim, stride, name, reuse=False, bias=None):

    with  tf.variable_scope(name, reuse=reuse):

        bottleneck_dim = round(expansion_ratio * input.get_shape().as_list()[-1])
        net = ConvBlock(input, bottleneck_dim, kernel_size=1, Use_bias=bias)

        net = depthwise_conv(net, name=name)
        net = instance_norm(net,scope='1')
        net = lrelu(net)

        net = Conv2D(net, output_dim, kernel_size=1)
        net = instance_norm(net,scope='2')

        if (int(input.get_shape().as_list()[-1]) == output_dim) and stride == 1:
            net = input + net

        return net

def depthwise_conv(input, k_h=3, k_w=3, channel_multiplier=1, strides=[1, 1, 1, 1],padding='VALID', name='dwise_conv', bias = True):

    input = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
    with tf.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]
        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],regularizer=None,initializer=tf.contrib.layers.variance_scaling_initializer())
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None, name=name, data_format=None)
        if bias:
            biases = tf.get_variable('bias', [in_channel * channel_multiplier],initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
        return conv


def Conv2D(inputs, filters, kernel_size=3, strides=1, padding='VALID', Use_bias = None):
    if kernel_size == 3 and strides == 1:
        inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
    if kernel_size == 7 and strides == 1:
        inputs = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], mode="REFLECT")
    if strides == 2:
        inputs = tf.pad(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="REFLECT")
    return tf.contrib.layers.conv2d(
        inputs,
        num_outputs=filters,
        kernel_size=kernel_size,
        stride=strides,
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
        biases_initializer= Use_bias,
        normalizer_fn=None,
        activation_fn=None,
        padding=padding)


def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if (kernel - stride) % 2 == 0 :
            pad_top = pad
            pad_bottom = pad
            pad_left = pad
            pad_right = pad

        else :
            pad_top = pad
            pad_bottom = kernel - stride - pad_top
            pad_left = pad
            pad_right = kernel - stride - pad_left

        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)


        return x

def deconv(x, channels, kernel=4, stride=2, use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()
        output_shape = [x_shape[0], x_shape[1]*stride, x_shape[2]*stride, channels]
        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')

            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                           strides=stride, padding='SAME', use_bias=use_bias)

        return x


##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, scope='resblock_0'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)


        return x + x_init

##################################################################################
# Sampling
##################################################################################

def flatten(x) :
    return tf.layers.flatten(x)

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

def sigmoid(x) :
    return tf.sigmoid(x)

##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)


def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)


def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):

        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

##################################################################################
# Loss function
##################################################################################

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss

def L2_loss(x,y):
    size = tf.size(x)
    return tf.nn.l2_loss(x-y)* 2 / tf.to_float(size)

def Huber_loss(x,y):
    return tf.losses.huber_loss(x,y)


def discriminator_loss(loss_func, real, real_gray, fake, real_blur):
    real_loss = 0
    fake_loss = 0
    real_gray_loss = 0
    real_blur_loss = 0

    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)
        real_blur_loss = tf.reduce_mean(real_blur)

    if loss_func == 'lsgan' :
        real_loss = tf.reduce_mean(tf.square(real - 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))
        real_gray_loss = tf.reduce_mean(tf.square(real_gray))
        real_blur_loss = tf.reduce_mean(tf.square(real_blur))

    if loss_func == 'gan' or loss_func == 'dragan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))
        real_blur_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real_blur), logits=real_blur))

    if loss_func == 'hinge':
        real_loss = tf.reduce_mean(relu(1.0 - real))
        fake_loss = tf.reduce_mean(relu(1.0 + fake))
        real_blur_loss = tf.reduce_mean(relu(1.0 + real_blur))

    loss = real_loss + fake_loss + real_blur_loss + real_gray_loss

    return loss


def generator_loss(loss_func, fake):
    fake_loss = 0

    if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.square(fake - 1.0))

    if loss_func == 'gan' or loss_func == 'dragan':
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge':
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss

    return loss




def gram(x):
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)

def con_loss(vgg, real, fake):

    vgg.build(real)
    real_feature_map = vgg.conv4_4_no_activation

    vgg.build(fake)
    fake_feature_map = vgg.conv4_4_no_activation

    loss = L1_loss(real_feature_map, fake_feature_map)

    return loss

def style_loss(style, fake):
    return L1_loss(gram(style), gram(fake))

def colmake_one_shot_iterator_loss(con, fake):
    con = rgb2yuv(con)
    fake = rgb2yuv(fake)

    return  L1_loss(con[:,:,:,0], fake[:,:,:,0]) + Huber_loss(con[:,:,:,1],fake[:,:,:,1]) + Huber_loss(con[:,:,:,2],fake[:,:,:,2])


def col_loss(con, fake):
    con = rgb2yuv(con)
    fake = rgb2yuv(fake)

    return  L1_loss(con[:,:,:,0], fake[:,:,:,0]) + Huber_loss(con[:,:,:,1],fake[:,:,:,1]) + Huber_loss(con[:,:,:,2],fake[:,:,:,2])


##Total variation loss常被用在损失函数里的正则项，可以起到平滑图像，去除鬼影，消除噪声的作用
def total_variation_loss(inputs):

    dh = inputs[:, :-1, ...] - inputs[:, 1:, ...]
    dw = inputs[:, :, :-1, ...] - inputs[:, :, 1:, ...]
    size_dh = tf.size(dh, out_type=tf.float32)
    size_dw = tf.size(dw, out_type=tf.float32)
    return tf.nn.l2_loss(dh) / size_dh + tf.nn.l2_loss(dw) / size_dw


def rgb2yuv(rgb):
    rgb = (rgb + 1.0)/2.0

    return tf.image.rgb_to_yuv(rgb)
