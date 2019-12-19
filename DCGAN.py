"""
write by qianqianjun
2019.12.20
DCGAN 网络架构实现
"""
from generator import Generator
from discriminater import Discriminator
import tensorflow as tf
class DCGAN(object):
    def __init__(self,hps):
        """
        建立一个DCGAN的网络架构
        :param hps:  网络的所有超参数的集合
        """
        g_channels=hps.g_channels
        d_channels=hps.d_channels
        self._batch_size=hps.batch_size
        self._init_conv_size=hps.init_conv_size
        self._z_dim=hps.z_dim
        self._img_size=hps.img_size
        self._generator=Generator(g_channels,self._init_conv_size)
        self._discriminator=Discriminator(d_channels)

    def build(self):
        """
        构建整个计算图
        :return:
        """
        # 创建随机向量和图片的占位符
        self._z_placeholder=tf.placeholder(tf.float32,
                                           (self._batch_size,self._z_dim))
        self._img_placeholder=tf.placeholder(tf.float32,
                                             (self._batch_size,
                                              self._img_size,
                                              self._img_size,1))
        # 将随机向量输入生成器生成图片
        generated_imgs=self._generator(self._z_placeholder,training=True)

        # 将来生成的图片经过判别器来得到 生成图像的logits
        fake_img_logits=self._discriminator(
            generated_imgs,training=True
        )
        # 将真实的图片经过判别器得到真实图像的 logits
        real_img_logits=self._discriminator(
            self._img_placeholder,training=True
        )

        """
        定义损失函数
        包括生成器的损失函数和判别器的损失函数。
        生成器的目的是使得生成图像经过判别器之后尽量被判断为真的
        判别器的目的是使得生成器生成的图像被判断为假的，同时真实图像经过判别器要被判断为真的
        """

        ## 生层器的损失函数，只需要使得假的图片被判断为真即可
        fake_is_real_loss=tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.ones([self._batch_size],dtype=tf.int64),
                logits=fake_img_logits
            )
        )

        ## 判别器的损失函数，只需要使得生成的图像被判断为假的，真实的图像被判断为真的即可
        # 真的被判断为真的：
        real_is_real_loss=tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.ones([self._batch_size],dtype=tf.int64),
                logits=real_img_logits
            )
        )
        # 假的被判断为假的：
        fake_is_fake_loss=tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.zeros([self._batch_size],dtype=tf.int64),
                logits=fake_img_logits
            )
        )

        # 将损失函数集中管理：
        tf.add_to_collection('g_losses',fake_is_real_loss)
        tf.add_to_collection('d_losses',real_is_real_loss)
        tf.add_to_collection('d_losses',fake_is_fake_loss)

        loss={
            'g':tf.add_n(tf.get_collection('g_losses'),name='total_g_loss'),
            'd':tf.add_n(tf.get_collection('d_losses'),name='total_d_loss')
        }
        return (self._z_placeholder,self._img_placeholder,generated_imgs,loss)
    def build_train_op(self,losses,learning_rate,beta1):
        """
        定义训练过程
        :param losses:  损失函数集合
        :param learning_rate:  学习率
        :param beta1:  指数衰减率估计
        :return:
        """
        g_opt=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)
        d_opt=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)

        g_opt_op=g_opt.minimize(
            losses['g'],
            var_list=self._generator.variables
        )

        d_opt_op=d_opt.minimize(
            losses['d'],
            var_list=self._discriminator.variables
        )

        with tf.control_dependencies([g_opt_op,d_opt_op]):
            return tf.no_op(name='train')
