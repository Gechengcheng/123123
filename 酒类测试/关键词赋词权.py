import numpy as np
import pandas as pd
import re
import collections
# 准备用户词典
from gensim import corpora, models

import jieba.posseg as pseg
import jieba
jieba.load_userdict(r"baijiu.txt")
jieba.load_userdict(r"purple.txt")

# 读取研究的一级类目优质文档的数据
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


# word2bow
def product_bow(series):
    """
    将文章列表进行doc2bow转换
    """
    text_jiu_list = [i for i in series.values]
    dictionary = corpora.Dictionary(text_jiu_list)
    doc_bow = [dictionary.doc2bow(text) for text in text_jiu_list]
    return doc_bow


# tfidf  关键词提取
def create_word_id_dict(text_token_list):
    text_list = [i for i in text_token_list.values]
    dictionary = corpora.Dictionary(text_list)
    word_id_dict = dict(zip(dictionary.token2id.values(),dictionary.token2id.keys()))
    return word_id_dict


def tfidf_weight(doc_bow):
    """
    计算三级类目tfidf值
    """
    tfidf = models.TfidfModel(doc_bow)
    tfidf_vectors = tfidf[doc_bow]
    return tfidf_vectors


def sorted_tfidf(tfidf_vectors, text_token_list):
    """
    排序并转换id为word(word,tfidf)并按照tfidf降序
    """
    L_third_sorted_tfidf = []
    for i in range(len(tfidf_vectors)):
        tuple_list_temp = [x for x in tfidf_vectors[i]]
        dict_temp = dict(tuple_list_temp)
        temp_doc = sorted(dict_temp.items(), key=lambda x: x[1], reverse=True)  # 按值降序排列
        word_id_dict = create_word_id_dict(text_token_list)
        tfidf_sorted_tuple_list = [(word_id_dict.get(x[0]), x[1]) for x in temp_doc]
        L_third_sorted_tfidf.append(tfidf_sorted_tuple_list)
    return L_third_sorted_tfidf


def state_word_tfidf(L_third_sorted_tfidf):
    keyword_cate = collections.defaultdict(list)
    for text in L_third_sorted_tfidf:
        for k, v in text:
            keyword_cate[k].append(v)
    max_tfidf_dict = dict([(key, np.max(keyword_cate[key])) for key in keyword_cate.keys()])
    max_tfidf_dict = dict(sorted(max_tfidf_dict.items(), key=lambda x: x[1], reverse=True))
    return max_tfidf_dict


# lda 关键词提取
def create_word_dict(text_token_list):
    text_list = [i for i in text_token_list.values]
    dictionary = corpora.Dictionary(text_list)
    return dictionary


def lda_weight(doc_bow, text_token_list, topic_num, word_num):
    """
    返回lda模型的主题
    topic_num:主题数
    word_num:主题词个数
    """
    dictionary = create_word_dict(text_token_list)
    lda = models.ldamodel.LdaModel(corpus=doc_bow, id2word=dictionary, num_topics=topic_num, minimum_probability=0)  # slow
    lda_topics = [x[1] for x in lda.print_topics(num_topics=-1, num_words=word_num)]
    return lda_topics


def extract_weight_word(lda_topics):
    temp_topic_list = [x.split(" + ") for x in lda_topics]
    some_topic = [(x.split("*")[1][1:-1], float((x.split("*")[0]))) for j in temp_topic_list for x in j]
    keyword_cate = collections.defaultdict(list)
    for k, v in some_topic:
        keyword_cate[k].append(v)
    max_lda_dict = dict([(key, np.max(keyword_cate[key])) for key in keyword_cate.keys()])
    max_lda_dict = dict(sorted(max_lda_dict.items(), key=lambda x: x[1], reverse=True))
    return max_lda_dict  # df_keyword后续


# 关键词表生成
def df_keyword(max_dict):
    df_keys = pd.DataFrame(max_dict.keys(), columns=["keys"])
    df_values = pd.DataFrame(max_dict.values(), columns=["values"])
    df_keyword_weight = pd.concat([df_keys, df_values], axis=1)
    df_keyword_weight["scale_weight"] = df_keyword_weight["values"].apply(lambda x: (x - np.min(df_keyword_weight["values"])) / (np.max(df_keyword_weight["values"]) - np.min(df_keyword_weight["values"])))
    df_keyword_weight.drop("values", axis=1, inplace=True)
    df_keyword_weight = df_keyword_weight.loc[(df_keyword_weight["scale_weight"] > 0) & (df_keyword_weight["scale_weight"] < 1)]
    df_keyword_weight.reset_index(drop=True, inplace=True)
    return df_keyword_weight


# 测试 三级ID为9435.0 白酒为例
# test_白酒_keyword_weight 测试
if __name__ == '__main__':
    # 白酒数据准备
    df_wine_baijiu = locate_third_cate(9435.0)
    description_concat_baijiu = data_prepare(df_wine_baijiu["main_title"], df_wine_baijiu["sub_title"], df_wine_baijiu["description"])
    description_concat_baijiu_clean = description_concat_baijiu["corpus_prepare"].apply(clean_sent)
    description_concat_baijiu_clean = description_concat_baijiu_clean.apply(tokenization)  # slow待优化
    # word2bow
    baijiu_bow = product_bow(description_concat_baijiu_clean)
    # tfidf关键词
    tfidf_baijiu = tfidf_weight(baijiu_bow)
    baijiu_desc_tfidf = sorted_tfidf(tfidf_baijiu, description_concat_baijiu_clean)  # 2分钟
    baijiu_max_tfidf_dict = state_word_tfidf(baijiu_desc_tfidf)
    baijiu_df_keyword_weight = df_keyword(baijiu_max_tfidf_dict)
    baijiu_df_keyword_weight.to_csv(r"baijiu_tfidf_weight.csv")
    # lda关键词
    baijiu_lda_topics = lda_weight(baijiu_bow, description_concat_baijiu_clean, 20, 25)
    baijiu_max_lda_dict = extract_weight_word(baijiu_lda_topics)
    baijiu_df_keyword_weight_lda = df_keyword(baijiu_max_lda_dict)
    baijiu_df_keyword_weight_lda.to_csv(r"baijiu_lda_weight.csv")

