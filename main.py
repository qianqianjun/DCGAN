"""
write by qianqianjun
2019.12.20
运行GAN进行训练的入口文件。
"""
import os
import tensorflow as tf
from train_argparse import hps
from dataset_loader import train_images
from data_provider import MnistData
from DCGAN import DCGAN
from utils import combine_imgs

# 创建生成结果目录
output_dir='./out'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# 创建DCGAN
dcgan=DCGAN(hps)
# 加载Mnist 数据集
mnist_data=MnistData(train_images,hps.z_dim,hps.img_size)
# 构建计算图模型
z_placeholder,img_placeholder,generated_imgs,losses=dcgan.build()

# 构建训练过程模型
train_op=dcgan.build_train_op(losses,hps.learning_rate,hps.beta1)

# 开始进行训练～ ：
init_op=tf.global_variables_initializer()
train_steps=200
with tf.Session() as sess:
    sess.run(init_op)
    for step in range(train_steps):
        batch_imgs,batch_z=mnist_data.next_batch(hps.batch_size)
        fetches=[train_op,losses['g'],losses['d']]
        should_sample=(step+1) %100 ==0
        # 如果到了该保存中间结果的步骤，则run 的时候在 fetches 中加上生成的图像
        if should_sample:
            fetches+= [generated_imgs]
        output_values=sess.run(
            fetches,feed_dict={
                z_placeholder:batch_z,
                img_placeholder:batch_imgs,
            }
        )
        _,g_loss_val,d_loss_val=output_values[0:3]
        # 打印训练过程的损失情况
        if (step+1) %200==0:
            print('step: %4d , g_loss: %4.3f , d_loss: %4.3f' % (step, g_loss_val, d_loss_val))

        # 保存中间过程图片结果：
        if should_sample:
            gen_imgs_val=output_values[3]
            gen_img_path=os.path.join(output_dir,'%05d-gen.jpg' % (step+1))
            gt_img_path=os.path.join(output_dir,'%05d-gt.jpg' % (step+1))
            gen_img=combine_imgs(gen_imgs_val,hps.img_size)
            gt_img=combine_imgs(batch_imgs,hps.img_size)
            gen_img.save(gen_img_path)
            gt_img.save(gt_img_path)