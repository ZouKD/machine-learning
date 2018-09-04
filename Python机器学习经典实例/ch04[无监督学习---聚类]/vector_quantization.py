"""
用矢量量化压缩图片
    k-means聚类的主要应用之一就是矢量量化。简单来说，矢量量化就是“四舍五入（rounding-off）”
    的N维版本。在处理数字等一维数据时，会用四舍五入计数减少存储空间。例如，如果只需要精确
    到两位小数，那么不会直接存储23.7344545,而是用23.73来代替。如果不去关心小数部分，甚至
    可以直接存储24，这取决于我们的真实需求。
"""

import argparse

import numpy as np
from scipy import misc
from sklearn import cluster
from matplotlib import pyplot as plt

# 解析输入参数，把图片和每个像素被压缩的比特数传进去作为输入参数
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compress the input image using clustering' )
    parser.add_argument("--input-file", dest="input_file", required=True,
                        help="Input image")
    parser.add_argument("--num-bits", dest="num_bits", required=False,
                        type=int, help="Number of bits used to represent each pixel")
    return parser

# 压缩输入的图片
def compress_image(img, num_clusters):
    # 将输入的图片转换成（样本量，特征量）数组，以运行k-means聚类算法
    X = img.reshape((-1,1))

    # 对输入数据运行k-means
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=4, random_state=5)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_

    # 为每个数据配置离它最近的中心点，并转变为图片的形状
    input_image_compressed = np.choose(labels, centroids).reshape(img.shape)

    return input_image_compressed

# 画图
def plot_image(img, title):
    vmin = img.min()
    vmax = img.max()
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
    # plt.show()

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    input_file = args.input_file
    num_bits = args.num_bits

    if not 1 <= num_bits <= 8:
        raise TypeError('Number of bits should between 1 and 8')
    num_clusters = np.power(2, num_bits)

    # 打印压缩率
    compression_rate = round(100 * (8.0 - args.num_bits) / 8.0, 2)
    print("The size of the image will be reduced by a factor of", 8.0/args.num_bits)
    print("Compression rate = " + str(compression_rate) + '%')

    # 加载输入图片
    input_image = misc.imread(input_file, True).astype(np.uint8)

    # 显示原始图片
    plot_image(input_image, 'Original image')

    # 压缩图片
    input_image_compressed = compress_image(input_image, num_clusters)
    plot_image(input_image_compressed, 'Compressed image; compression rate = '
               + str(compression_rate) + '%')
    plt.show()








































