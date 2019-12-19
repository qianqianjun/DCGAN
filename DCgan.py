"""
数据准备
    1.图像数据
    2.随机向量
构建计算图
    1.生成器
    2.判别器
    3.DCGAN 架构编写
      1) 链接生成器和判别器
      2) 定义损失函数
      3) 定义train_op
训练过程
"""
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import os

output_dir='./local_run'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

def get_default_params():
    return tf.contrib.training.HParams(
        z_dim=100,
        init_conv_size=4,
        g_channels=[128,64,32,1],
        d_channels=[32,64,128,256],
        batch_size=128,
        learning_rate=0.002,
        betal=0.5,
        img_size=32
    )

hps=get_default_params()


# 实现数据的provider
# class MnistData(object):
#     def __init__(self,mnist_train,z_dim,img_size):
#         self._data=mnist_train
#         self._example_num=len(self._data)
#         self._z_data=np.random.standard_normal((self._example_num,z_dim))
#         self._indicator=0
#         self._resize_mnist(img_size)
#         self._random_shuffer()
#     def _random_shuffer(self):
#         p=np.random.permutation(self._example_num)
#         self._z_data=self._z_data[p]
#         self._data=self._data[p]
#
#     def _resize_mnist(self,img_size):
#         """
#         将图片转换为对应的size
#         :param img_size:
#         :return:
#         """
#         #data=np.asarray(self._data * 255,np.uint8)
#         data=np.reshape(self._data,(self._example_num,28,28))
#         new_data=[]
#         for i in range(self._example_num):
#             img=data[i]
#             img=Image.fromarray(img)
#             img=img.resize((img_size,img_size))
#             img=np.asarray(img)
#             img=img.reshape((img_size,img_size,1))
#             new_data.append(img)
#
#         new_data=np.asarray(new_data,dtype=np.float32)
#         new_data=new_data /127.5 -1
#         self._data=new_data
#
#     def next_batch(self,batch_size):
#         end_dicator=self._indicator+ batch_size
#         if end_dicator > self._example_num:
#             self._random_shuffer()
#             self._indicator=0
#             end_dicator=self._indicator+batch_size
#         batch_data=self._data[self._indicator:end_dicator]
#         batch_z=self._z_data[self._indicator:end_dicator]
#         self._indicator=end_dicator
#         return batch_data,batch_z


from dataset_loader import train_images
from data_provider import MnistData
mnist_data=MnistData(train_images,hps.z_dim,hps.img_size)
batch_data,batch_z=mnist_data.next_batch(5)
## 生成器部分

def conv2d_transpose(inputs,out_channel,name,training,with_bn_relu=True):
    """
    反卷积操作的封装。
    :param inputs:
    :param out_channel:
    :param name:
    :param training:
    :param with_bn_relu:
    :return:
    """
    with tf.variable_scope(name):
        conv2d_trans=tf.layers.conv2d_transpose(
            inputs,out_channel,[5,5],
            strides=(2,2),
            padding='SAME'
        )
        if with_bn_relu:
            bn=tf.layers.batch_normalization(conv2d_trans,training=training)
            return tf.nn.relu(bn)
        else:
            return conv2d_trans

# 编写生成器模型
class Generator(object):
    def __init__(self,channels,init_conv_size):
        self._channels=channels
        self._init_conv_size=init_conv_size
        self._resue=False

    def __call__(self, inputs,training):
        inputs=tf.convert_to_tensor(inputs)
        with tf.variable_scope('generator',reuse=self._resue):
            """
            random vector -> fc -> self._channels[0] * init_conv_size **2
            -> reshape-> [init_conv_size,init_conv_size,channels]
            """
            with tf.variable_scope('inputs_conv'):
                fc=tf.layers.dense(
                    inputs,
                    self._channels[0]* self._init_conv_size * self._init_conv_size
                )
                conv0=tf.reshape(fc,[-1,self._init_conv_size,
                                     self._init_conv_size,self._channels[0]])
                bn0=tf.layers.batch_normalization(conv0,training=training)
                relu0=tf.nn.relu(bn0)

            deconv_inputs=relu0
            for i in range(1,len(self._channels)):
                with_bn_relu=(i!=len(self._channels)-1)
                deconv_inputs=conv2d_transpose(
                    deconv_inputs,
                    self._channels[i],
                    "deconv-%d" % i,
                    training,
                    with_bn_relu)
            img_inputs=deconv_inputs
            with tf.variable_scope('generate_imgs'):
                # imgs value range [-1,1]
                imgs=tf.tanh(img_inputs,name='imgs')

        self._resue=True
        self.variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='generator')
        return imgs


# 定义判别层
def conv2d(inputs,output_channel,name,training):
    """
    :param inputs:
    :param output_channel:
    :param name:
    :param training:
    :return:
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
        self._channels=channels
        self._reuse=False
    def __call__(self, inputs, training):
        inputs=tf.convert_to_tensor(inputs,dtype=tf.float32)
        conv_inputs=inputs
        with tf.variable_scope('discriminator',reuse=self._reuse):
            for i in range(len(self._channels)):
                conv_inputs=conv2d(conv_inputs,self._channels[i],
                                   'conv-%d' %i,
                                   training)
            fc_inputs=conv_inputs
            with tf.variable_scope('fc'):
                flatten=tf.layers.flatten(fc_inputs)
                logits=tf.layers.dense(flatten,2,name='logits')
        self._reuse=True
        self.variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope='discriminator')
        return logits



class DCGAN(object):
    def __init__(self,hps):
        g_channels=hps.g_channels
        d_channels=hps.d_channels
        self._batch_size=hps.batch_size
        self._init_conv_size=hps.init_conv_size
        self._z_dim=hps.z_dim
        self._img_size=hps.img_size
        self._generator=Generator(g_channels,self._init_conv_size)
        self._discriminator=Discriminator(d_channels)

    def build(self):
        "build 整个计算图"
        self._z_placeholder=tf.placeholder(tf.float32,
                                           (self._batch_size,self._z_dim))
        self._img_placeholder=tf.placeholder(
            tf.float32,(
                self._batch_size,
                self._img_size,
                self._img_size,1))
        generatd_imgs=self._generator(self._z_placeholder,training=True)

        fake_img_logits=self._discriminator(
            generatd_imgs,training=True
        )
        real_img_logits=self._discriminator(
            self._img_placeholder,training=True
        )

        # 生成器loss
        loss_on_fake_to_real=tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.ones([self._batch_size],dtype=tf.int64),
                logits=fake_img_logits
            )
        )
        # 判别器loss
        loss_on_fake_to_fake=tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.zeros([self._batch_size],dtype=tf.int64),
                logits=fake_img_logits
            )
        )
        loss_on_real_to_real=tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.ones([self._batch_size],dtype=tf.int64),
                logits=real_img_logits
            )
        )

        tf.add_to_collection('g_losses',loss_on_fake_to_real)
        tf.add_to_collection('d_losses',loss_on_fake_to_fake)
        tf.add_to_collection('d_losses',loss_on_real_to_real)

        loss={
            'g':tf.add_n(tf.get_collection('g_losses'),name='total_g_loss'),
            'd':tf.add_n(tf.get_collection('d_losses'),name='total_d_loss')
        }
        return (self._z_placeholder,self._img_placeholder,generatd_imgs,loss)
    def build_train_op(self,losses,learning_rate,beta1):
        g_opt=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)
        d_opt=tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=beta1)

        g_opt_op=g_opt.minimize(
            losses['g'],var_list=self._generator.variables
        )
        d_opt_op=g_opt.minimize(
            losses['d'],var_list=self._discriminator.variables
        )

        with tf.control_dependencies([g_opt_op,d_opt_op]):
            return tf.no_op(name='train')


# 调用DCGAN
dcgan=DCGAN(hps)
z_placeholder,img_placeholder,generated_imgs,losses=dcgan.build()
train_op=dcgan.build_train_op(losses,hps.learning_rate,hps.betal)


# 训练流程过程构建
def combine_imgs(batch_images,img_size,rows=8,cols=16):
    #batch_img: [batch_size,img_size,img_size,1]
    result_big_img=[]
    for i in range(rows):
        row_imgs=[]
        for j in range(cols):
            img=batch_images[cols*i+j]
            img=img.reshape((img_size,img_size))
            # 反归一化
            img=(img+1) * 127.5
            row_imgs.append(img)
        row_imgs=np.hstack(row_imgs)
        result_big_img.append(row_imgs)
    result_big_img=np.vstack(result_big_img)
    result_big_img=np.asarray(result_big_img,np.uint8)
    result_big_img=Image.fromarray(result_big_img)
    return result_big_img

init_op=tf.global_variables_initializer()
train_steps=1000
with tf.Session() as sess:
    sess.run(init_op)
    for step in range(train_steps):
        batch_imgs,batch_z=mnist_data.next_batch(hps.batch_size)
        fetches=[train_op,losses['g'],losses['d']]
        should_sample=(step+1) %100 ==0
        if should_sample:
            fetches+= [generated_imgs]
        output_values=sess.run(
            fetches,feed_dict={
                z_placeholder:batch_z,
                img_placeholder:batch_imgs,
            }
        )
        _,g_loss_val,d_loss_val=output_values[0:3]
        if (step+1) %200==0:
            print('step: %4d , g_loss: %4.3f , d_loss: %4.3f' % (step, g_loss_val, d_loss_val))
        if should_sample:
            gen_imgs_val=output_values[3]
            gen_img_path=os.path.join(output_dir,'%05d-gen.jpg' % (step+1))
            gt_img_path=os.path.join(output_dir,'%05d-gt.jpg' % (step+1))
            gen_img=combine_imgs(gen_imgs_val,hps.img_size)
            gt_img=combine_imgs(batch_imgs,hps.img_size)
            gen_img.save(gen_img_path)
            gt_img.save(gt_img_path)