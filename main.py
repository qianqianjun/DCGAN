import os
import tensorflow as tf
from train_argparse import hps
from dataset_loader import train_images
from data_provider import MnistData
from DCGAN import DCGAN
from utils import combine_imgs

output_dir='./out'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
dcgan=DCGAN(hps)
z_placeholder,img_placeholder,generated_imgs,losses=dcgan.build()
train_op=dcgan.build_train_op(losses,hps.learning_rate,hps.beta1)
init_op=tf.global_variables_initializer()
train_steps=200
mnist_data=MnistData(train_images,hps.z_dim,hps.img_size)
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