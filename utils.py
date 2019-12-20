"""
write by qianqianjun
2019,12,20

工具文件
这里使用了 numpy 的一些维度变换，如果不清楚可以参考博客：
https://blog.csdn.net/qq_38863413/article/details/103526645
"""
import numpy as np
from PIL import Image
def combine_imgs(batch_images,img_size,rows=8,cols=16):
    """
    用于在训练过程中展示一批数据（将一批图像拼接成一张大图）
    :param batch_images:  批次图像数据
    :param img_size:  图像大小
    :param rows:  一共有多行。
    :param cols:  一行放置多少图片
    :return:  返回拼接之后的大图
    """
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