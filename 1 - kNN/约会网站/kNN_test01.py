# -*- coding: UTF-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import operator


def classify0(inX, data_set, labels, k):
    """
    函数说明:kNN算法,分类器
    Parameters:
        inX - 用于分类的数据(测试集)
        data_set - 用于训练的数据(训练集)
        labes - 分类标签
        k - kNN算法参数,选择距离最小的k个点
    Returns:
        sorted_class_count[0][0] - 分类结果
    """
    #numpy函数shape[0]返回data_set的行数
    data_set_size = data_set.shape[0]
    #在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共data_setSize次(纵向)
    diff_mat = np.tile(inX, (data_set_size, 1)) - data_set
    #二维特征相减后平方
    sq_diff_mat = diff_mat ** 2
    #sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sq_distances = sq_diff_mat.sum(axis=1)
    #开方，计算出距离
    distances = sq_distances**0.5
    #返回distances中元素从小到大排序后的索引值
    sorted_dist_indices = distances.argsort()
    #定义一个记录类别次数的字典
    class_count = {}
    for i in range(k):
        #取出前k个元素的类别
        vote_I_label = labels[sorted_dist_indices[i]]
        #dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        #计算类别次数
        class_count[vote_I_label] = class_count.get(vote_I_label,0) + 1
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    #reverse降序排序字典
    sorted_class_count = sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    #返回次数最多的类别,即所要分类的类别
    return sorted_class_count[0][0]

def file2matrix(filename):
    """
    函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力
    Parameters:
        filename - 文件名
    Returns:
        return_mat - 特征矩阵
        class_lable_vector - 分类Label向量
    """
    # 打开文件，指定编码
    fr = open(filename,'r',encoding = 'utf-8')
    # 读取文件内容
    array_olines = fr.readlines()
    # 针对有BOM的UTF-8文本，应该去掉BOM，否则后面会引发错误。
    array_olines[0]=array_olines[0].lstrip('\ufeff')
    # 得到文件行数
    number_of_lines = len(array_olines)
    # 创建返回的NumPy矩阵
    return_mat = np.zeros((number_of_lines, 3))
    # 返回的分类标签向量
    class_lable_vector = []
    # 行的索引值
    index = 0
    
    for line in array_olines:
        #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        #使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        list_from_line = line.split('\t')
        #将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        return_mat[index, :] = list_from_line[0:3]
        #根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if list_from_line[-1] == 'didntLike':
            class_lable_vector.append(1)
        elif list_from_line[-1] == 'smallDoses':
            class_lable_vector.append(2)
        elif list_from_line[-1] == 'largeDoses':
            class_lable_vector.append(3)
        index += 1
    return return_mat, class_lable_vector

def showdatas(dating_data_mat, dating_labels):
    """
    函数说明:可视化数据
    Parameters:
        dating_data_mat - 特征矩阵
        dating_labels - 分类Label
    Returns:
        无
    """
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))
    number_of_lables = len(dating_labels)
    labels_colors = []
    for i in dating_labels:
        if i == 1:
            labels_colors.append('black')
        if i == 2:
            labels_colors.append('orange')
        if i == 3:
            labels_colors.append('red')

    #画出散点图,以dating_data_mat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=dating_data_mat[:,0], y=dating_data_mat[:,1], color=labels_colors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')  
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')  
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black') 

    #画出散点图,以dating_data_mat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=dating_data_mat[:,0], y=dating_data_mat[:,2], color=labels_colors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')  
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')  
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black') 

    #画出散点图,以dating_data_mat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=dating_data_mat[:,1], y=dating_data_mat[:,2], color=labels_colors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')  
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')  
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black') 
    #设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='largeDoses')
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    #显示图片
    plt.show()

def auto_norm(data_set):
    """
    函数说明:对数据进行归一化
    Parameters:
        data_set - 特征矩阵
    Returns:
        norm_data_set - 归一化后的特征矩阵
        ranges - 数据范围
        min_vals - 数据最小值
    """
    #获得数据的最小值
    min_vals = data_set.min(0)
    maxVals = data_set.max(0)
    #最大值和最小值的范围
    ranges = maxVals - min_vals
    #shape(data_set)返回data_set的矩阵行列数
    norm_data_set = np.zeros(np.shape(data_set))
    #返回data_set的行数
    m = data_set.shape[0]
    #原始值减去最小值
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    #除以最大和最小值的差,得到归一化数据
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    #返回归一化数据结果,数据范围,最小值
    return norm_data_set, ranges, min_vals

def dating_class_test():
    """
    函数说明:分类器测试函数
    取百分之十的数据作为测试数据，检测分类器的正确性
    Parameters:
        无
    Returns:
        无
    """
    #打开的文件名
    filename = "datingTestSet.txt"
    #将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    dating_data_mat, dating_labels = file2matrix(filename)
    #取所有数据的百分之十
    ho_ratio = 0.10
    #数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    #获得normMat的行数
    m = norm_mat.shape[0]
    #百分之十的测试数据的个数
    num_test_vecs = int(m * ho_ratio)
    #分类错误计数
    error_count = 0.0

    for i in range(num_test_vecs):
        #前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifier_result = classify0(norm_mat[i,:], norm_mat[num_test_vecs:m,:], 
            dating_labels[num_test_vecs:m], 4)
        print("分类结果:%s\t真实类别:%d" % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print("错误率:%f%%" %(error_count/float(num_test_vecs)*100))

def classify_person():
    """
    函数说明:通过输入一个人的三维特征,进行分类输出
    Parameters:
        无
    Returns:
        无
    """
    #输出结果
    result_list = ['讨厌','有些喜欢','非常喜欢']
    #三维特征用户输入
    precent_tats = float(input("玩视频游戏所耗时间百分比:"))
    ff_miles = float(input("每年获得的飞行常客里程数:"))
    icecream = float(input("每周消费的冰激淋公升数:"))
    #打开的文件名
    filename = "datingTestSet.txt"
    #打开并处理数据
    dating_data_mat, dating_labels = file2matrix(filename)
    #训练集归一化
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    #生成NumPy数组,测试集
    in_arr = np.array([ff_miles, precent_tats, icecream])
    #测试集归一化
    normin_arr = (in_arr - min_vals) / ranges
    #返回分类结果
    classifier_result = classify0(normin_arr, norm_mat, dating_labels, 3)
    #打印结果
    print("你可能%s这个人" % (result_list[classifier_result-1]))


if __name__ == '__main__':
    dating_class_test()


