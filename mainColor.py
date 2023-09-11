from sklearn.cluster import KMeans
import numpy as np
import warnings
import os
from PIL import Image
import argparse  # 1、导入argpase包
import tkinter as tk
from tkinter import filedialog


warnings.filterwarnings('ignore')


def parse_args():
    parse = argparse.ArgumentParser(description='maincolor')  
    parse.add_argument('--image', default='', type=str, help='添加想要解析的图片')  
    parse.add_argument('--count', default=5, type=int, help='预计提取出的主要颜色个数（默认前5个）')
    args = parse.parse_args() 
    return args

if __name__ == '__main__':
    
    root = tk.Tk()
    root.withdraw()

    img_file = filedialog.askopenfilename()
    #print()
    #print('\n获取的文件地址：', img_file)



    #args = parse_args()
    #img_file = str(input('请输入待分析图片的路径（需后缀）:'))# args.image
    #top = int(input('预计提取出主要颜色个数：')) # args.count
    top = 5
    
    #print("正在处理的图片是：", img_file)
    #print()
    # k-means中的k值，即选择几个中心点
    k = top

    # 读图片
    #img = io.imread(img_file)
    img = Image.open(img_file)

    width, height = img.size
 
    # print(width*(400/height))
    img = img.resize((int(width*(400/height)), 400))
    img = np.array(img)
    # 转换数据维度
    img_ori_shape = img.shape


    img1 = img.reshape((img_ori_shape[0] * img_ori_shape[1], img_ori_shape[2]))
    img_shape = img1.shape

    # 获取图片色彩层数
    n_channels = img_shape[1]

    estimator = KMeans(n_clusters=k, max_iter=4000, init='k-means++', n_init=50)  # 构造聚类器
    estimator.fit(img1)  # 聚类
    centroids = estimator.cluster_centers_  # 获取聚类中心

    colorLabels = list(estimator.labels_)
    colorInfo = {}
    for center_index in range(k):
        colorRatio = colorLabels.count(center_index)/len(colorLabels)
        colorInfo[colorRatio] = centroids[center_index]

    # 根据比例从高至第低排序
    '''bonus: 颜色占比排序'''
    colorInfo = [(k,colorInfo[k]) for k in sorted(colorInfo.keys(), reverse=True)] 
    percentage = []
    for idx, color in enumerate(colorInfo):
        percentage.append(color[0])
        #print('Top-{} 主成分 RGB 值：'.format(idx+1), '(R={} G={} B={})'.format(format(color[1][0], '.1f'), format(color[1][1], '.1f'), format(color[1][2], '.1f')) , ' \n             所占比重：', int(color[0]*100),'%')
        #print()
        #print()
        
    # 根据中心点生成一个矩阵
    result = []
    result_width = 200
    result_height_per_center = 80
    for center_index in range(k):

        #result = np.hstack((result,np.full((result_width * result_height_per_center, n_channels), colorInfo[center_index][1], dtype=int)))
        result.append(np.full((int(result_width * result_height_per_center), n_channels), colorInfo[center_index][1], dtype=int))
        #print(result)

    result = np.array(result)
    result = result.reshape((result_height_per_center * k, result_width, n_channels))

    img = np.resize(img,(result.shape[0], img.shape[1]*(result.shape[0]//img.shape[0]),n_channels))
    final = np.concatenate((img,result),axis=1)
    #print(np.array(result).shape)
    # print(img.shape)
    # print(result.shape)

    # im=Image.fromarray(np.uint8(result))
    # im.show()
    # im=Image.fromarray(np.uint8(img))
    # im.show()
    im=Image.fromarray(np.uint8(final))
    im.show()
    '''
    ok =True
    while ok:
        delete = str(input("是否删除原图（yes/no）："))
        if delete == "yes" or delete == "no":
            ok = False
        else:
            print("输入不合法")
    if delete == "yes":
        os.remove(img_file)
    else:
        pass
    '''
    # 保存图片
    #io.imsave(os.path.splitext(img_file)[0] + '_result.bmp', result)
