# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
import re
import jieba
from jieba import analyse
from snownlp import SnowNLP
from wordcloud import WordCloud  # 词云展示模块
from wordcloud import ImageColorGenerator
import PIL.Image as image  # 图像处理模块
import matplotlib.pyplot as plt  # 图像展示模块
import matplotlib.font_manager as fm  # 中文处理模块
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

os.chdir(r"D:\gechengcheng3\Desktop\hotpot")
raw_comment = pd.read_csv(r"hotpot_gcc_comments.csv", sep='\t', usecols=[1, 2, 3], nrows=5000) #先取出两千条分析
# EDA
print("评论内容为空的占比：{:.5f}".format(sum(raw_comment["comment_content"].isnull())/len(raw_comment)))
print("\n")
print("评论时间为空的占比：{:.5f}".format(sum(raw_comment["comment_create_tm"].isnull())/len(raw_comment)))
print("\n")
print("评论内容无效占比：{:.5f}".format(sum(raw_comment["comment_content"] == "此用户未填写评价内容")/len(raw_comment)))
# 评论内容和评论时间为空的记录条数并不一致，下沉查看，时间不为空时内容为空，确定是否只保留最新的评论创建时间
tm_cont = raw_comment.loc[raw_comment["comment_create_tm"].isnull(), ["comment_create_tm", "comment_content"]]  # 时间为空时内容为空
cont_tm = raw_comment.loc[raw_comment["comment_content"].isnull(), ["comment_create_tm", "comment_content"]]   # 内容为空时时间不一定为空，可能是评编辑之后又删除了
# 评论内容清洗：取出无效评论 & 为空的评论
raw_comment_clear = raw_comment.loc[(raw_comment["comment_content"].notnull()) & (raw_comment["comment_content"] != "此用户未填写评价内容"), :].copy() # 或者的条件用 |
# 不同sku_id 的评论数
sku_id_num = raw_comment_clear['item_sku_id'].value_counts().sort_values(ascending=False)
# 分词，去停用词，提取关键字,使用snownlp来做情感分析，该库的训练就是以购物数据为样本进行训练的


# 计算情感倾向得分
def sentiments_analysis(sentence):  # 利用snowNLP包定义情感分析函数，输出得分：得分越靠近1视为肯定，靠近0视为否定
    sentence_score = SnowNLP(sentence).sentiments # 提取情感分析的得分
    return sentence_score


raw_comment_clear["sentence_score"] = raw_comment_clear["comment_content"].apply(sentiments_analysis)


def mappings(score):
    if score > 0.6:
        return 2
    elif score > 0.3:
        return 1
    else:
        return 0


raw_comment_clear["label"] = raw_comment_clear["sentence_score"].apply(mappings)
# 标签转化,阈值可能后期需要调整；0-0.3,0.6-1,0.3-0.6中性评价，暂时不予挖掘
raw_comment_clear.reset_index(range(len(raw_comment_clear)), drop=True, inplace=True)
raw_comment_clear["label"].value_counts().sort_values(ascending=False)  # 样本不平衡，正负样本4:1
# 语料分类
negative_corpus = raw_comment_clear.loc[raw_comment_clear["label"] == 2, :]["comment_content"]
positive_corpus = raw_comment_clear.loc[raw_comment_clear["label"] == 0, :]["comment_content"]
# 分词，去停用词，绘制词云图，正向评价+负向评价

pattern = re.compile(r'[^\u4e00-\u9fa5]+')
corpus_clean = []
for raw in raw_comment_clear["comment_content"]:
    raw = pattern.sub('', str(raw))
    raw = raw.strip()
    corpus_clean.append(raw)


# 去除标点
def eliminate_punctuation(sentence):
    return pattern.sub('', str(sentence))


raw_comment_clear["sent_clear"] = raw_comment_clear["comment_content"].apply(eliminate_punctuation)

# 分词
def cut_words(sentence):
    return " ".join(jieba.cut(sentence))


raw_comment_clear["cut_words"] = raw_comment_clear["sent_clear"].apply(cut_words)

# 使用哈工大停用词典
with open('stop_words.txt', "r", encoding='UTF-8') as f:
    stop_words = f.read().replace("\n", " ")
    stop_words = stop_words.split()  # 1598个停用词


def eliminate_stopwords(sentence):
    temp_list = [x for x in sentence.split() if x not in stop_words]
    return " ".join(temp_list)


raw_comment_clear["cut_words_filter"] = raw_comment_clear["cut_words"].apply(eliminate_stopwords)

tfidf_content_positive = []
# po_ne_dict = {0: "positive", 1: "negative"}
for content in raw_comment_clear.loc[raw_comment_clear["label"] == 2, :]["sent_clear"]:
    tags = jieba.analyse.extract_tags(" ".join(jieba.cut(content)))
    tfidf_content_positive.append(tags)

tfidf_content_negative = []
for content in raw_comment_clear.loc[raw_comment_clear["label"] == 0, :]["sent_clear"]:
    tags = jieba.analyse.extract_tags(" ".join(jieba.cut(content)))
    tfidf_content_negative.append(tags)


# 绘制词云图——positive
sentence = []
for sent in tfidf_content_negative:
    text = " ".join(sent)
    sentence.append(text)
text = " ".join(sentence)
mask = np.array(image.open('1.jpg'))  # 图片背景参考形状
wc = WordCloud(
    background_color="white",  # 背景颜色
    max_words=150,  # 最多显示的词数
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
wc.to_file("negative.png")  # 输出一个png文件保存云图


# 绘制词云图——negative
sentence = []
for sent in tfidf_content_negative:
    text = " ".join(sent)
    sentence.append(text)
text = " ".join(sentence)
mask = np.array(image.open('1.jpg'))  # 图片背景参考形状
wc = WordCloud(
    background_color="white",  # 背景颜色
    max_words=150,  # 最多显示的词数
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
wc.to_file("negative.png")  # 输出一个png文件保存云图

# 手动建立评价词典，分词时引入即可，纳入小火锅评价知识库
positive_corpus.to_csv("positive_corpus.csv")
negative_corpus.to_csv("negative_corpus.csv")
# 建立 评论词汇表

# 修正ing


# LDA 主题模型,报错警告
corpus = list(raw_comment_clear.loc[raw_comment_clear["label"] == 2, :]["comment_content"])
cntVector = CountVectorizer(stop_words=stop_words)
cntTf = cntVector.fit_transform(corpus)
print(cntTf.todense())
print(cntTf.todense().shape)  # 个是稀疏矩阵 (1601, 3670)

print(cntTf.toarray())
print(cntTf.toarray().shape)  # (1601, 3670),1601篇标签为2的文档，3670个词的词典，后面的是词频
# 训练模型
lda = LatentDirichletAllocation(n_topics=5,
                                learning_offset=50.,
                                random_state=10)
docres = lda.fit_transform(cntTf)
print(docres)  # (1601,5) 每篇评论的主题分布
"""
[[0.06668627 0.06668738 0.06668634 0.06668297 0.73325704]
 [0.06668588 0.06668652 0.73325923 0.06668282 0.06668556]
 [0.05011276 0.05000463 0.7998741  0.05000368 0.05000484]
 ...
 [0.10000254 0.10000267 0.59998596 0.10000633 0.1000025 ]
 [0.06666993 0.0666704  0.73332034 0.06666942 0.06666991]
 [0.10000657 0.10000685 0.10000651 0.10000546 0.59997461]]"""
print(lda.components_)  # （5，3670）五个主题，每个主题中词的分布
"""
[[0.20554041 2.39852273 0.20508539 ... 0.20511002 1.08421393 0.20483628]
 [0.2050625  0.20628425 0.20541994 ... 0.2048558  0.20481801 1.23533281]
 [0.20511342 0.20522484 0.2048121  ... 0.20481967 0.20498903 0.20440061]
 [0.20459005 0.20469956 0.20412783 ... 0.20418647 0.20484234 0.20457494]
 [1.15812111 0.20529283 2.08540494 ... 2.39917349 0.2056573  0.20469423]]
 """

# 如何确定是对应主题的含义？——答：根据每一个主题下的关键词和语句自行确定
# 设置主题，口味，分量，物流，价格，服务，num_topics=5
# https://blog.csdn.net/Yellow_python/article/details/83097994?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param
from gensim import corpora, models
import jieba.posseg as jp, jieba
# 文本集
texts = [
    '美国教练坦言，没输给中国女排，是输给了郎平' * 99,
    '美国无缘四强，听听主教练的评价' * 99,
    '中国女排晋级世锦赛四强，全面解析主教练郎平的执教艺术' * 99,
    '为什么越来越多的人买MPV，而放弃SUV？跑一趟长途就知道了' * 99,
    '跑了长途才知道，SUV和轿车之间的差距' * 99,
    '家用的轿车买什么好' * 99]
# 分词过滤条件
jieba.add_word('四强', 9, 'n')
flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 词性
stopwords = ('没', '就', '知道', '是', '才', '听听', '坦言', '全面', '越来越', '评价', '放弃', '人')  # 停词
# 分词
words_ls = []
for text in texts:
    words = [w.word for w in jp.cut(text) if w.flag in flags and w.word not in stopwords]
    words_ls.append(words)
# 构造词典
dictionary = corpora.Dictionary(words_ls)
# 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
corpus = [dictionary.doc2bow(words) for words in words_ls]
# lda模型，num_topics设置主题的个数
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=2)
# 打印所有主题，每个主题显示5个词
for topic in lda.print_topics(num_words=5):
    print(topic)
# 主题推断
print(lda.inference(corpus))
# 主题推断
for e, values in enumerate(lda.inference(corpus)[0]):
    print(texts[e])
    for ee, value in enumerate(values):
        print('\t主题%d推断值%.2f' % (ee, value))

# 词和主题的关系
word_id = dictionary.doc2idx(['长途'])[0]
for i in lda.get_term_topics(word_id):
    print('【长途】与【主题%d】的关系值：%.2f%%' % (i[0], i[1]*100))

# jiagu文本聚类
import jiagu
docs = raw_comment_clear["comment_content"][100:200].copy()
cluster = jiagu.text_cluster(docs, 5)
print(cluster)


