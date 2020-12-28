import os
import pandas as pd
import numpy as np
import re
import jieba
from collections import defaultdict
from gensim import corpora
from jieba import analyse

os.chdir(r"D:\gechengcheng3\Desktop\hotpot")
raw_data = pd.read_csv(r"gcc_hotpot_search_0721_20200721185640.csv", sep='\t')
pd.set_option("display.max_row", 50)

# 统计总的搜索次数和非空搜索占比
# 见jupyter notebook

# 清洗文本并保存在corpus_clean.txt中
corpus_clean = []
fil = re.compile(r'[^0-9.\u4e00-\u9fa5]+')
for raw in raw_data.loc[:, "keyword"]:
    raw = fil.sub('', str(raw))
    raw = raw.strip()
    corpus_clean.append(raw)
    #print(raw)
with open(r"corpus_clean.txt", 'w') as f:
    for sent in corpus_clean:
        f.write(sent)
        f.write("\n")

"""
#step2:结巴分词
#2.1补充并对比分词粒度调整
"""
f = open("corpus_clean.txt", "r")   # 设置文件对象
string = f.readlines()     # 将txt文件的所有内容读入到字符串string中
f.close()
string=string[12:].copy() # 去除nan


def fenci(data):
    cut_words = map(lambda s: list(jieba.cut(s)), data)
    return list(cut_words)


fenci_list = fenci(string)
sum_ci = 0
for i in fenci_list:
    sum_ci += len(i)
print("加入分词词典之前：", sum_ci)

jieba.load_userdict("userdict.txt")
fenci_list = fenci(string)
sum_ci=0
for i in fenci_list:
    sum_ci += len(i)
print("加入分词词典之后：",sum_ci)

"""
删除搜索关键字中的停用词
"""
f=open('stop_words.txt', "r", encoding='UTF-8')
stop_words_string = f.readlines()
f.close()
for i in range(len(stop_words_string)):
    stop_words_string[i] = stop_words_string[i][:-1]
stop_words_string.append("\n")
for search_list in fenci_list:
    for word in search_list:
        if word in stop_words_string:
            search_list.remove(word)
sum_ci = 0
for i in fenci_list:
    sum_ci += len(i)
print("加入分词词典且去停用词之后：", sum_ci)
"""
加入分词词典之前： 2360309
加入分词词典之后： 2208889
加入分词词典且去停用词之后： 1634455
"""

"""
step3:词频统计
"""
frequency = defaultdict(int)
for text in fenci_list:
    for token in text:
        frequency[token] += 1
print(len(frequency))
frequency_sort = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
frequency_sort = pd.DataFrame(frequency_sort, columns=["词语", "词频"])
print("打印词频:", frequency_sort)

# try1:搜索关键字，词聚类，不考虑不同类用户特性，一把抓
# 训练词向量
# 使用 gensim 中的 word2vec训练词向量
model = Word2Vec(fenci_list, size=200, workers=5, sg=1)  # 生成词向量为200维，考虑上下5个单词共10个单词，采用sg=1的方法也就是skip-gram
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
model.train(fenci_list, total_examples=1, epochs=1)
model.wv.most_similar("份量")
model.wv.most_similar("味道")
model.wv.most_similar(u"海底捞")  # 默认10个,加u不加u没有区别
model.wv.most_similar("自嗨锅", topn=5)  # 找出最相似的前20个词

# 建立词向量词典
w2c = {}
w2c_list = []
for word in list(frequency.keys()):  # dict_keys
    try:
        w2c[word] = model.wv[word]
        w2c_list.append(model.wv[word])
    except:
        pass
# test
vec1 = model.wv["自嗨锅"]
vec2 = model.wv["羊肉"]
vec3 = model.wv["重庆"]  # 此处必要要加wv,否则会发出警告

# 根据词向量进行词聚类——k-means
data = np.array(w2c_list)
k_means = KMeans(n_clusters=5, random_state=10).fit(data)

# 输出聚类中心，解决不可迭代问题
cluster_centers_5 = k_means.cluster_centers_  # array
clus_set_shape_modify = []
for cc in cluster_centers_5:
    #print(cc.shape)
    cc = cc[np.newaxis, :]
    #print(cc.shape)
    clus_set_shape_modify.append(cc)

# 解决完毕，开始迭代找五个类别的相似词
clus_set_all = [model.wv.most_similar(clus_set_shape_modify[i], topn=30) for i in range(5)]

# 绘制词云图可视化搜索关键词
from wordcloud import WordCloud  # 词云展示模块
from wordcloud import ImageColorGenerator
import PIL.Image as image  # 图像处理模块
import matplotlib.pyplot as plt  # 图像展示模块
import matplotlib.font_manager as fm  # 中文处理模块
sentence = []
for sent in fenci_list:
    text = " ".join(sent)
    sentence.append(text)
text = " ".join(sentence)
mask = np.array(image.open('1.jpg'))  # 图片背景参考形状
wc = WordCloud(
    background_color="white",  # 背景颜色
    max_words=100,  # 最多显示的词数
    mask=mask,  # 设置图片的背景
    max_font_size=80,  # 最大的字符
    random_state=10,  # 设置有多少种随机生成状态，即有多少种配色方案
    font_path=r'C:/Windows/Fonts/simkai.ttf'  # 中文处理，用系统自带的字体
).generate(text)  # generate 只能处理str文本不能处理list文本
# 对词云图各项参数进行调整，使词云图更美观
my_font = fm.FontProperties(fname=r'C:/Windows/Fonts/simkai.ttf')  # 词云字体设置
image_colors = ImageColorGenerator(mask)  # 基于彩色图像的颜色生成器 提取背景图片颜色
wc.recolor(color_func=image_colors)  # 将词云颜色设置为背景图的颜色
plt.axis("off")  # 为云图去掉坐标轴
plt.imshow(wc, interpolation="bilinear")  # 开始画图
wc.to_file("search_keywords.png")  # 输出一个png文件保存云图