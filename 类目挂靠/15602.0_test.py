
import pandas as pd
import re
import jieba
import jieba.posseg as pseg #带有词性标注的切词
from gensim.models import CoherenceModel
import collections
from gensim import corpora, models, similarities
import jieba.posseg as pseg
# 准备用户词典
jieba.load_userdict(r"baijiu.txt")
jieba.load_userdict(r"purple.txt")
from collections import defaultdict

df_wine = pd.read_csv(r"wine_0814_gcc_copy_20200831165920.csv", sep="\t", encoding="utf-8")


# 数据准备
def locate_third_cate(float_num):
    """
    function:locate 对应三级类目的优质文档的文本
    :param float_num: 传入三级类目代码
    :return: 数据框
    """
    df_wine_cate = df_wine.loc[df_wine["three_category"] == float_num].copy()  # 白酒，后面要加.0
    return df_wine_cate


def title_concat(series1, series2):
    """
    function:实现合并main_title和sub_title,重复的保留一个，不重复的加总
    series1 main_title
    series2 sub_title
    return list
    """
    list_title_concat = []
    for i, j in zip(series1, series2):
        if i == j and pd.notnull(j):
            list_title_concat.append(i)
        elif i != j and pd.isnull(j):
            list_title_concat.append(i)
        elif i != j and pd.isnull(i):
            list_title_concat.append(j)
        elif i != j and pd.notnull(i):
            list_title_concat.append(str(i)+str(j))
        else:
            list_title_concat.append(0)
    return list_title_concat


# title合并content
def description_concat(list1, series1):
    list_description_content = []
    for i, j in zip(list1, series1):
        list_description_content.append(str(i)+str(j))
    return list_description_content


def data_prepare(series1, series2, series3):
    """
    封装title和description的步骤,返回文档数据框
    """
    title_concat1 = title_concat(series1, series2)
    description_concat1 = description_concat(title_concat1, series3)
    description_concat_df = pd.DataFrame(description_concat1, columns=["corpus_prepare"])
    return description_concat_df


# 清洗文本+分词+停词过滤+无关词性过滤+单个字过滤
fil = re.compile(r'[^\u4e00-\u9fa5]+')

# clean_sent和tokennize函数都是针对小单元进行的，可以用于apply里面


def clean_sent(raw_sent):
    raw_sent = fil.sub('', str(raw_sent))
    return raw_sent


# 停用词获取
def StopwordsList(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    stopwords = list(set(stopwords))
    return stopwords


stopwordlist = StopwordsList(r"停用词.txt")


stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
# 切词+停用词过滤
stop_flag_dict = dict(zip(stop_flag, stop_flag))
stopwordlist_dict = dict(zip(stopwordlist, stopwordlist))


def tokenization(data):
    """
    词性过滤，停用词过滤
    """
    result = []
    words = pseg.cut(data)
    for word, flag in words:
        flag1 = stop_flag_dict.get(flag, None)
        word1 = stopwordlist_dict.get(word, None)
#         if flag not in stop_flag and word not in stopwordlist:
        if pd.isnull(word1) or pd.isnull(flag1):
            if len(str(word)) >= 2:
                result.append(word)
    return result

# 训练优质文档的 bow& dict
def product_bow(series):
    """
    将优质文章列表进行doc2bow转换
    """
    text_jiu_list = [i for i in series.values]
    dictionary = corpora.Dictionary(text_jiu_list)
    doc_bow = [dictionary.doc2bow(text) for text in text_jiu_list]
    return doc_bow,dictionary

def tfidf_weight(doc_bow):
    """
    计算三级类目tfidf值
    """
    tfidf = models.TfidfModel(doc_bow)
    tfidf_vectors = tfidf[doc_bow]
    index = similarities.MatrixSimilarity(tfidf_vectors)
    return tfidf,tfidf_vectors,index

def lda_topic_dictribution(doc_bow,dictionary,topic_num,word_num):
    """
    返回lda模型的主题
    topic_num:主题数
    word_num:主题词个数
    """
    lda = models.ldamodel.LdaModel(corpus=doc_bow, id2word=dictionary, num_topics=topic_num,minimum_probability=0) #slow
    lda_topics = [x[1] for x in lda.print_topics(num_topics=-1, num_words=word_num)]
    lda_vectors = lda[doc_bow]
    index = similarities.MatrixSimilarity(lda[doc_bow])
    return lda,lda_topics,lda_vectors,index

def topic_sims(ldamodel,new_bows,pijiu_index):
    list_lda_sims = []
    new_bow_lda_ls = [ldamodel[new_vec] for new_vec in new_bows]
    sims = [pijiu_index[new_vec_tfidf] for new_vec_tfidf in new_bow_lda_ls]
    for sim in sims:
        array2dict = dict(zip(range(len(sim)),list(sim)))
        temp = sorted(array2dict.items(),key=lambda x: x[1],reverse=True)[:10]
        temp_max = temp[0]
        if temp_max[1] < 0.9:
            list_lda_sims.append(None)
        else:
            list_lda_sims.append(temp)
    return list_lda_sims

# 舆情文档的 bow
def yuqing_bow(dictionary,yuqing_series):
    text_yuqing_list = [i for i in yuqing_series.values]
    doc_bow = [dictionary.doc2bow(text) for text in text_yuqing_list]
    return doc_bow


if __name__ == '__main__':
    df_wine_pijiu = locate_third_cate(15602.0)
    description_concat_pijiu = data_prepare(df_wine_pijiu["main_title"], df_wine_pijiu["sub_title"], df_wine_pijiu["description"])
    description_concat_pijiu_clean = description_concat_pijiu["corpus_prepare"].apply(clean_sent)
    description_concat_pijiu_clean = description_concat_pijiu_clean.apply(tokenization)  # time
    # 啤酒 doc_bow
    pijiu_bow, pijiu_dictionary = product_bow(description_concat_pijiu_clean)
    # 啤酒优质文档 模型，tfidf结果，index相似度 计算准备
    tfidf, pijiu_tfidf, index = tfidf_weight(pijiu_bow)
    # 根据 coherence score确定主题个数
    for i in range(3, 7):
        lda = lda_topic_dictribution(pijiu_bow, pijiu_dictionary, i, 25)[0]
        goodcm = CoherenceModel(model=lda, texts=description_concat_pijiu_clean, dictionary=pijiu_dictionary,
                                coherence='c_v')
        print(goodcm.get_coherence())
    # 五个主题的coherence score连贯性得分最高
    lda_pijiu, pijiu_lda_topics, pijiu_lda_vectors, pijiu_index = lda_topic_dictribution(pijiu_bow, pijiu_dictionary, 5,25)
    # 微博舆情测试数据清洗
    data_weibo = pd.read_excel(r"280_fdm_31337512.xlsx")  # weibocontent
    weibo_clean = data_weibo["weibocontent"].apply(clean_sent)
    weibo_clean_token = weibo_clean.apply(tokenization)

