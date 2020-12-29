from ops import *
from utils import *
from glob import glob
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np

class AnimeGAN(object) :
    def __init__(self, sess, args):
        self.model_name = 'AnimeGAN'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag
        self.epoch = args.epoch
        self.init_epoch = args.init_epoch # args.epoch // 20
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.init_lr = args.init_lr
        self.d_lr = args.d_lr
        self.g_lr = args.g_lr

        self.ch = args.ch

        """ Weight """
        self.g_adv_weight = args.g_adv_weight
        self.d_adv_weight = args.d_adv_weight
        self.con_weight = args.con_weight
        self.sty_weight = args.sty_weight
        self.color_weight = args.color_weight

        self.ld = args.ld


        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_critic = args.n_critic
        self.sn = args.sn

        self.img_size = args.img_size
        self.img_ch = args.img_ch


        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/train_real'))
        self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/anime_style'))
        self.trainB_gray_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/anime_gray'))
        self.trainB_smooth_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/anime_smooth'))

        self.dataset_num = max(len(self.trainA_dataset), len(self.trainB_dataset))
        self.vgg = Vgg19()
        print()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# init_epoch : ", self.init_epoch)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : 8")

        print()

        print("##### Discriminator #####")
        print("# the number of discriminator layer : ", self.n_dis)
        print("# the number of critic : ", self.n_critic)
        print("# spectral normalization : ", self.sn)

        print()

    ##################################################################################
    # Generator
    ##################################################################################
    def generator(self, x_init, reuse=False, scope="generator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :

            x_init = ConvBlock(x_init,64)
            x_init = ConvBlock(x_init,64)
            x_init = Down_Conv(x_init,128)

            x_init = ConvBlock(x_init,128)
            x_init = DSConv(x_init, 128)
            x_init = Down_Conv(x_init,256)
            x_init = ConvBlock(x_init,256)


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
            x_init = ConvBlock(x_init,64)
            x_init = ConvBlock(x_init, 64)

            output = Conv2D(x_init, 3, 1)

            return tf.tanh(output)


    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, reuse=False, scope="discriminator"):
        channel = self.ch // 2
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x_init, channel, kernel=3, stride=1, pad=1, use_bias=False, sn=self.sn, scope='conv_0')
            x = lrelu(x, 0.2)

            for i in range(1, self.n_dis):
                x = conv(x, channel * 2, kernel=3, stride=2, pad=1, use_bias=False, sn=self.sn, scope='conv_s2_' + str(i))
                x = lrelu(x, 0.2)

                x = conv(x, channel * 4, kernel=3, stride=1, pad=1, use_bias=False, sn=self.sn, scope='conv_s1_' + str(i))
                x = instance_norm(x, scope='ins_norm_' + str(i))
                x = lrelu(x, 0.2)

                channel = channel * 2

            x = conv(x, channel * 2, kernel=3, stride=1, pad=1, use_bias=False, sn=self.sn, scope='last_conv')
            x = instance_norm(x, scope='last_ins_norm')
            x = lrelu(x, 0.2)

            x = conv(x, channels=1, kernel=3, stride=1, pad=1, use_bias=False, sn=self.sn, scope='D_logit')

            return x

    ##################################################################################
    # Model
    ##################################################################################
    def gradient_panalty(self, real, fake, scope="discriminator"):
        if self.gan_type.__contains__('dragan') :
            eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.)
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

            fake = real + 0.5 * x_std * eps

        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = real + alpha * (fake - real)

        logit = self.discriminator(interpolated, reuse=True, scope=scope)


        grad = tf.gradients(logit, interpolated)[0] # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=1) # l2 norm

        GP = 0
        # WGAN - LP
        if self.gan_type.__contains__('lp'):
            GP = self.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif self.gan_type.__contains__('gp') or self.gan_type == 'dragan' :
            GP = self.ld * tf.reduce_mean(tf.square(grad_norm - 1.))


        return GP

    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Input Image"""
        Image_Data_Class = ImageData(self.img_size, self.img_ch, self.augment_flag)

        trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
        trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)
        trainB_gray = tf.data.Dataset.from_tensor_slices(self.trainB_gray_dataset)
        trainB_smooth = tf.data.Dataset.from_tensor_slices(self.trainB_smooth_dataset)

        gpu_device = '/gpu:0'
        trainA = trainA.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))
        trainB = trainB.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))
        trainB_gray = trainB_gray.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))
        trainB_smooth = trainB_smooth.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, self.batch_size))

        trainA_iterator = trainA.make_one_shot_iterator()
        trainB_iterator = trainB.make_one_shot_iterator()
        trainB_gray_iterator = trainB_gray.make_one_shot_iterator()
        trainB_smooth_iterator = trainB_smooth.make_one_shot_iterator()


        self.real_A = trainA_iterator.get_next()
        self.real_B = trainB_iterator.get_next()
        self.real_B_gray = trainB_gray_iterator.get_next()
        self.real_B_smooth = trainB_smooth_iterator.get_next()

        self.test_real_A = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_real_A')


        """ Define Generator, Discriminator """
        #真实图通过生成器获得假图
        self.fake_B = self.generator(self.real_A)


        #风格图通过判别器获得风格图
        real_B_logit = self.discriminator(self.real_B)
        real_B_gray_logit = self.discriminator(self.real_B_gray, reuse=True)

        fake_B_logit = self.discriminator(self.fake_B, reuse=True)
        real_B_smooth_logit = self.discriminator(self.real_B_smooth, reuse=True)


        """ Define Loss """
        if self.gan_type.__contains__('gp') or self.gan_type.__contains__('lp') or self.gan_type.__contains__('dragan') :
            GP = self.gradient_panalty(real=self.real_B, fake=self.fake_B)
        else :
            GP = 0.0

        # init pharse
        init_c_loss = con_loss(self.vgg,self.real_A, self.fake_B)
        init_loss = self.con_weight * init_c_loss

        self.init_loss = init_loss

        content_loss = con_loss(self.vgg,self.real_A,self.fake_B)
        gray_style_loss = style_loss(self.real_B_gray,self.fake_B)
        color_loss = col_loss(self.real_A,self.fake_B)

        g_loss = self.g_adv_weight * generator_loss(self.gan_type, fake_B_logit)
        d_loss = self.d_adv_weight * discriminator_loss(self.gan_type, real_B_logit, real_B_gray_logit,fake_B_logit, real_B_smooth_logit) + GP

        t_loss = self.con_weight * content_loss + self.sty_weight * gray_style_loss + self.color_weight * color_loss

        self.Generator_loss = g_loss + t_loss
        self.Discriminator_loss = d_loss


        """ Result Image """
        self.test_fake_B = self.generator(self.test_real_A, reuse=True)

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]

        self.init_optim = tf.train.AdamOptimizer(self.init_loss, beta1=0.5, beta2=0.999).minimize(self.init_loss, var_list=G_vars)
        self.G_optim = tf.train.AdamOptimizer(self.g_lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.d_lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=D_vars)


        """" Summary """
        self.G_loss = tf.summary.scalar("Generator_loss", self.Generator_loss)
        self.D_loss = tf.summary.scalar("Discriminator_loss", self.Discriminator_loss)

        self.G_gan = tf.summary.scalar("G_gan", g_loss)
        self.G_vgg = tf.summary.scalar("G_vgg", t_loss)
        self.G_init_loss = tf.summary.scalar("G_init", init_loss)

        self.G_init_loss_merge = tf.summary.merge([self.G_init_loss])
        self.G_loss_merge = tf.summary.merge([self.G_loss, self.G_gan, self.G_vgg, self.G_init_loss])
        self.D_loss_merge = tf.summary.merge([self.D_loss])


    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)


        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        # start_time = time.time()
        past_g_loss = -1.
        lr = self.init_lr
        for epoch in range(start_epoch, self.epoch):
            if self.decay_flag :
                lr = self.init_lr * pow(0.5, epoch // self.decay_epoch)

            for idx in range(start_batch_id, self.iteration):

                train_feed_dict = {
                    self.lr : lr
                }

                if epoch < self.init_epoch :
                    # Init G
                    start_time = time.time()
                    real_A_images, fake_B_images, _, G_init_loss, summary_str = self.sess.run([self.real_A, self.fake_B,
                                                                             self.init_optim,
                                                                             self.init_loss, self.G_init_loss_merge], feed_dict = train_feed_dict)
                    self.writer.add_summary(summary_str, counter)
                    print("Epoch: [%3d] [%5d/%5d] time: %4.4f G_init_loss: %.8f" % (epoch, idx, self.iteration, time.time() - start_time, G_init_loss))

                else :
                    # Update D
                    _, d_loss, summary_str = self.sess.run([self.D_optim, self.Discriminator_loss, self.D_loss_merge], feed_dict = train_feed_dict)
                    self.writer.add_summary(summary_str, counter)

                    # Update G
                    g_loss = None
                    if (counter - 1) % self.n_critic == 0 :
                        real_A_images, fake_B_images, _, g_loss, summary_str = self.sess.run([self.real_A, self.fake_B,
                                                                                              self.G_optim,
                                                                                              self.Generator_loss, self.G_loss_merge], feed_dict = train_feed_dict)
                        self.writer.add_summary(summary_str, counter)
                        past_g_loss = g_loss

                    if g_loss == None:
                        g_loss = past_g_loss
                    start_time = time.time()
                    print("Epoch: [%3d] [%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (epoch, idx, self.iteration, time.time() - start_time, d_loss, g_loss))

                # display training status
                counter += 1


                if np.mod(idx+1, self.print_freq) == 0 :
                    save_images(real_A_images, [self.batch_size, 1],
                                './{}/real_A_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                    save_images(fake_B_images, [self.batch_size, 1],
                                './{}/fake_B_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                if np.mod(idx + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)



            start_batch_id = 0

            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        n_dis = str(self.n_dis) + 'dis'
        return "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(self.model_name, self.dataset_name,
                                                         self.gan_type, n_dis,
                                                         self.n_critic, self.sn,
                                                        int(self.g_adv_weight), int(self.d_adv_weight),
                                                        int(self.con_weight), int(self.sty_weight),
                                                        int(self.color_weight))

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir) # checkpoint file information

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path) # first line
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file))
            image_path = os.path.join(self.result_dir,'{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.test_fake_B, feed_dict = {self.test_real_A : sample_image})
            save_images(fake_img, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '../..' + os.path.sep + sample_file), self.img_size, self.img_size))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                '../..' + os.path.sep + image_path), self.img_size, self.img_size))
            index.write("</tr>")

        index.close()