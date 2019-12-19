"""
create by qianqianjun
2019.12.19
"""
import os
import struct
import numpy as np

def load_mnist(path,train=True):
    """
    加载mnist 数据集的函数
    :param path:  数据集的位置
    :param train:  是否加载训练数据，是返回train 用的image和lable，否则返回test用的images和label
    :return: 返回训练或者测试用的images 和 labels 
    """
    def get_urls(files,type='train'):
        """
        获取训练数据或者测试数据的二进制文件地址
        :param files:  读取的数据集目录文件列表
        :param type:  训练或者测试标识
        :return:  返回二进制文件的完整地址
        """
        images_path = None
        labels_path = None
        for file in files:
            if file.find(type) != -1:
                if file.find("images") != -1:
                    images_path = os.path.join(path, file)
                else:
                    labels_path = os.path.join(path, file)

        if images_path == None or labels_path == None:
            raise Exception("请检查数据集！")
        return images_path,labels_path
    def load_data_and_label(data_path,label_path):
        """
        加载训练或者测试数据的lable 和 data
        :param data_path:  训练或者测试图片数据的二进制文件地址
        :param label_path:  训练或者测试label数据的二进制文件地址
        :return:  返回读取的图片 和 label 的 ndarray 数组
        """
        images = None
        labels = None
        with open(label_path,'rb') as label_file:
            struct.unpack('>II', label_file.read(8))
            labels=np.fromfile(label_file,dtype=np.uint8)
        with open(data_path,'rb') as img_file:
            struct.unpack('>IIII', img_file.read(16))
            images=np.fromfile(img_file,dtype=np.uint8).reshape(len(labels),784)
        return images,labels

    # 查看数据集文件夹中有多少文件。
    files = os.listdir(path)
    if train:
        data_path,label_path=get_urls(files,type='train')
        return load_data_and_label(data_path,label_path)
    else:
        data_path,label_path=get_urls(files,type='t10k')
        return load_data_and_label(data_path, label_path)

# 读取训练用的图片数据和训练用的labels 标签
train_images,train_labels=load_mnist("./MNIST",train=True)
# 读取测试用的图片数据和测试用的labels 标签
test_images,test_labels=load_mnist("./MNIST",train=False)