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
from dataset_loader import train_images
from data_provider import MnistData
mnist_data=MnistData(train_images,hps.z_dim,hps.img_size)
batch_data,batch_z=mnist_data.next_batch(5)
## 生成器部分
from generator import Generator
# 判别器部分
from discriminater import Discriminator
# DCGAN
from DCGAN import DCGAN
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
train_steps=200
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