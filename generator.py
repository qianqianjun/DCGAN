"""
write by qianqianjun
2019.12.19
生成器模型实现
"""
import tensorflow as tf
def conv2d_transpose(inputs,output_channel,name,training,with_bn_relu=True):
    """
    反卷积的封装
    :param inputs:
    :param output_channel: 输出通道数目
    :param name: 名字
    :param training: bool类型 ，指示是否在训练
    :param with_bn_relu: 是否需要使用 batch_normalization
    :return: 反卷积之后的矩阵
    """
    with tf.variable_scope(name):
        conv2d_trains=tf.layers.conv2d_transpose(
            inputs,output_channel,[5,5],
            strides=(2,2),
            padding='SAME'
        )
        if with_bn_relu:
            bn=tf.layers.batch_normalization(conv2d_trains,training=training)
            return tf.nn.relu(bn)
        else:
            return conv2d_trains

class Generator(object):
    def __init__(self,channels,init_conv_size):
        """
        创建生成器模型
        :param channels: 生成器反卷积过程中使用的通道数 数组
        :param init_conv_size:  使用的卷积核大小
        """
        self._channels=channels
        self._init_conv_size=init_conv_size
        self._reuse=False
    def __call__(self, inputs,training):
        """
        一个魔法函数，用来将对象当函数使用
        :param inputs: 输入的随机向量矩阵，shape 为 【batch_size ,z_dim]
        :param training:  是否是训练过程
        :return: 返回生成的图像
        """
        inputs=tf.convert_to_tensor(inputs)
        with tf.variable_scope('generator'):
            """
            下面代码实现的转换是： random vector-> fc全连接层-> 
            self.channels[0] * self._init_conv_size **2 ->
            reshpe -> [init_conv_size,init_conv_size,self.channels[0] ]
            """
            with tf.variable_scope("input_conv"):
                fc=tf.layers.dense(
                    inputs,
                    self._channels[0] * self._init_conv_size **2
                )
                conv0=tf.reshape(fc,[-1,self._init_conv_size,
                                     self._init_conv_size,self._channels[0]])

                bn0=tf.layers.batch_normalization(conv0,training=training)
                relu0=tf.nn.relu(bn0)

            # 经过全连接和BN归一化和 relu 激活，可以看做是某一个卷积层的输出
            # 下面就可以进行反卷积操作了。
            deconv_inputs=relu0
            # 构建 decoder 网络层
            for i in range(1,len(self._channels)-1):
                with_bn_relu=(i!=len(self._channels)-1)
                deconv_inputs=conv2d_transpose(
                    deconv_inputs,
                    self._channels[i],
                    "deconv-%d" % i,
                    training,
                    with_bn_relu=with_bn_relu)
            img_inputs=deconv_inputs
            with tf.variable_scope('generate_imgs'):
                imgs=tf.tanh(img_inputs,name='imgs')

        self._reuse=True
        self.variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope='generator')
        return imgs