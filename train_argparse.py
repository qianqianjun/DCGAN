"""
write by qianqianjun
2019.12.20
命令行参数解释程序
如果不清楚可以参考博客：
https://blog.csdn.net/qq_38863413/article/details/103305449
"""
import argparse
parser=argparse.ArgumentParser()
parser.description="指定DCGAN网络在训练时候的超参数，使用help命令获取详细的帮助"
parser.add_argument("--batch_size",type=int,default=128,help="训练时候的批次大小，默认是128")
parser.add_argument("--learning_rate",type=float,default=0.002,help="训练时候的学习率，默认是0.002")
parser.add_argument("--img_size",type=int,default=32,help="生成图片的大小（和训练图片的大小保持一致）")
parser.add_argument("--z_dim",type=int,default=100,help="输入生成器的随机向量的大小，默认是100")
parser.add_argument("--g_channels",type=list,default=[128,64,32,1],help="生成器的通道数目变化列表，用于构建生成器结构")
parser.add_argument("--d_channels",type=list,default=[32,64,128,256],help="判别器的通道树木变化列表，用来构建判别器")
parser.add_argument("--init_conv_size",type=int,default=4,help="随机向量z经过全连接之后进行reshape 生成三维矩阵的初始边长，默认是 4 ")
parser.add_argument("--beta1",type=float,default=0.5,help="AdamOptimizer 指数衰减率估计，默认是0.5")

hps=parser.parse_args()