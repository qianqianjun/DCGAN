"""
write by qianqianjun
2019.12.20
判别器简单实现
"""
import tensorflow as tf
def conv2d(inputs,output_channel,name,training):
    """
    卷积操作的封装
    :param inputs: 输入的图像或者feature map
    :param output_channel:  输出feature map 的channel 数目
    :param name:  varibale_scope 名称
    :param training:  是否是训练过程。
    :return:  返回经过卷积层之后的结果
    """
    def leaky_relu(x,leak=0.2,name=''):
        return tf.maximum(x,x*leak,name=name)

    with tf.variable_scope(name):
        conv2d_output=tf.layers.conv2d(
            inputs,output_channel,
            [5,5],strides=(2,2),
            padding='SAME'
        )
        bn=tf.layers.batch_normalization(conv2d_output,training=training)
        return leaky_relu(bn,name='outputs')

class Discriminator(object):
    def __init__(self,channels):
        """
        创建判别器模型结构
        :param channels:  输出通道数目
        """
        self._channels=channels
        self._reuse=False
    def __call__(self,inputs,training):
        """
        使用判别器输出判别的结果，
        :param inputs:  输入的batch_images data
        :param training:  是否在训练。
        :return:
        """
        inputs=tf.convert_to_tensor(inputs,dtype=tf.float32)
        conv_inputs=inputs
        with tf.variable_scope('discriminator',reuse=self._reuse):
            # 根据卷积通道数组来建立卷积神经网络结构：
            for i in range(len(self._channels)):
                conv_inputs=conv2d(conv_inputs,self._channels[i],
                                   'conv-%d'%i,
                                   training=training)
            fc_inputs=conv_inputs
            # 将卷积神经网络输出的 feature map 展平并进行全连接。
            with tf.variable_scope('fc'):
                flatten=tf.layers.flatten(fc_inputs)
                # 全连接输出大小为 2
                # 其实可以理解为一个分类的问题，真图片还是假图片，一共两类。
                logits=tf.layers.dense(flatten,2,name='logits')
        self._reuse=True
        self.variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope='discriminator')
        return logits