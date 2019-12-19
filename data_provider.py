"""
write by qianqianjun
2019.12.20
"""
import numpy as np
from PIL import Image
class MnistData(object):
    def __init__(self,images_data,z_dim,img_size):
        """
        建立一个data provider
        :param images_data:  传进来的图像数据的集合
        :param z_dim:  生成器输入的随机向量的长度
        :param img_size:  传进来的图像的大小
        """
        self._data=images_data
        self.images_num=len(self._data)
        # 生成随机向量的矩阵，为每一张图像都生成一个随机向量。
        self._z_data=np.random.standard_normal((self.images_num,z_dim))
        self._offset=0
        self.init_mnist(img_size)
        self.random_shuffer()

    def random_shuffer(self):
        """
        数据集进行打乱操作，防止模型学习到训练数据之间的顺序性质
        :return:
        """
        p=np.random.permutation(self.images_num)
        self._z_data=self._z_data[p]
        self._data=self._data[p]

    def init_mnist(self,img_size):
        """
        调整数据集到指定的shape
        :param img_size: 指定大小的边长
        :return:
        """
        # 将训练数据进行resize，使其成为图片
        data=np.reshape(self._data,(self.images_num,28,28))
        new_data=[]
        for i in range(self.images_num):
            img=data[i]
            # 使用PIL 进行图像缩放变换
            img=Image.fromarray(img)
            img=img.resize((img_size,img_size))
            img=np.asarray(img)
            # 将图片转换为有通道的形式方便训练（3维矩阵，只有一个通道）
            img=img.reshape((img_size,img_size,1))
            new_data.append(img)
        # 将列表转换为 ndarray
        new_data=np.asarray(new_data,dtype=np.float32)
        # 对图像数据进行归一化，方便训练
        new_data=new_data / 127.5 -1
        # 更新数据
        self._data=new_data
    def next_batch(self,batch_size):
        """
        用来分批次的取数据
        :param batch_size:  每一批取数据的个数
        :return:  返回一批数据和一批随机向量
        """
        if batch_size> self.images_num:
            raise Exception("batch size is more than train images amount!")
        end_offset=self._offset+batch_size
        if end_offset >self.images_num:
            self.random_shuffer()
            self._offset=0
            end_offset=self._offset+batch_size

        # 取出一批数据和一批随机向量。
        batch_data=self._data[self._offset:end_offset]
        batch_z=self._z_data[self._offset:end_offset]
        self._offset=end_offset
        return batch_data,batch_z