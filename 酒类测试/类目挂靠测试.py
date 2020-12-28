import pandas as pd
import numpy as np
import jieba.posseg as pseg
from gensim import corpora, models, similarities
import heapq
import re
pd.set_option("display.max_columns", 50)

# 构建类目映射字典
three_cate_map_df = pd.read_csv(r"three_cate_0819_map_copy_20200819093641.csv", sep="\t")
dict_three = dict(zip(three_cate_map_df["item_third_cate_name"], three_cate_map_df["item_third_cate_cd"]))
dict_value_key = dict(zip(three_cate_map_df["item_third_cate_cd"], three_cate_map_df["item_third_cate_name"]))

# 读取优质文档数据
data = pd.read_csv(r"wine_0814_gcc_20200814102855.csv", sep="\t")
# 内容字段：main_title，sub_title,description,label标签字段，three_category
data.dropna(subset=['description'], inplace=True)  # 删除指定列为空值的行，subset=[]，必须写成这种格式否则会报错
data.drop_duplicates(inplace=True)  # 删除重复行
data.dropna(inplace=True, how="all", axis=0)
data.drop(data.loc[(data["main_title"].isnull()) & (data["sub_title"].isnull()) & (data["description"].isnull())].index, inplace=True)
data.reset_index(drop=True, inplace=True)
# 采样5万条建模，挂靠5000条舆情文档
df_sample_wine = data.sample(n=10000, replace=False, axis=0, random_state=123)
df_sample_wine.reset_index(drop=True, inplace=True)

# 实现合并main_title和sub_title,重复的保留一个，不重复的加总
L_title_concat = []
for i, j in zip(df_sample_wine["main_title"], df_sample_wine["sub_title"]):
    if i == j and pd.notnull(j):
        L_title_concat.append(i)
    elif i != j and pd.isnull(j):
        L_title_concat.append(i)
    elif i != j and pd.isnull(i):
        L_title_concat.append(j)
    elif i != j and pd.notnull(i):
        L_title_concat.append(str(i)+str(j))
    else:
        L_title_concat.append(0)

# title合并content
L_wine_content = []
for i, j in zip(L_title_concat, df_sample_wine["description"]):
    L_wine_content.append(str(i)+str(j))

# 清洗文本
fil = re.compile(r'[^\u4e00-\u9fa5]+')


def clean_sent(raw_sent):
    raw_sent = fil.sub('', str(raw_sent))
    return raw_sent


L_winecontent_concat_series = pd.Series(L_wine_content).apply(clean_sent)


# 停用词获取
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    stopwords = list(set(stopwords))
    return stopwords


stopwordlist = stopwordslist(r"停用词.txt")

stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']  # 数词，量词，介词，方位词...
# 切词+停用词过滤
stop_flag_dict = dict(zip(stop_flag, stop_flag))
stopwordlist_dict = dict(zip(stopwordlist, stopwordlist))


def tokenization(data):
    result = []
    words = pseg.cut(data)
    for word, flag in words:
        flag1 = stop_flag_dict.get(flag, None)
        word1 = stopwordlist_dict.get(word, None)
#         if flag not in stop_flag and word not in stopwordlist:
        if pd.isnull(word1) and pd.isnull(flag1):
            result.append(word)
    return result


winecontent_list = L_winecontent_concat_series.apply(tokenization)  # 执行2分钟左右，可以使用字典优化
text_wine_list = [i for i in winecontent_list.values]
dictionary_wine = corpora.Dictionary(text_wine_list)
doc_wine_vectors = [dictionary_wine.doc2bow(text) for text in text_wine_list]

lsi = models.LsiModel(doc_wine_vectors, id2word=dictionary_wine, num_topics=5)  # 将tfidf的结果作为LSI模型建立的材料
documents_wine_lsi = lsi[doc_wine_vectors]  # bow不是等维的，sklearn中是等维的，作为材料输出，计算的documents,documents是等维的，？？

index = similarities.MatrixSimilarity(documents_wine_lsi) # documents是等维的
index.save('test_10000.index')
index = similarities.MatrixSimilarity.load('test_10000.index')


def match_list_top10(series):
    L_index_match = []
    for sent in series.values:
        query_bow = dictionary_wine.doc2bow(sent)
        query_lsi = lsi[query_bow] # lsi 是以bow为输入材料进行计算的
        sims = index[query_lsi] # array，没有index
        sims = [x for x in sims]
        index_top_10 = map(sims.index, heapq.nlargest(10, sims))
        max_10_index = list(index_top_10)
        if [sims[i] for i in max_10_index][9] < 0.95:  # .mean(sims)
            L_index_match.append(None)
        else:
            L_index_match.append(max_10_index)
    return L_index_match


def cate_match_top1(index_match_list_of_list):
    three_match_list = []
    for index_list in index_match_list_of_list:
        if index_list != None:
            match_list = [df_sample_wine.iloc[i,3] for i in index_list]
            count_tuple_list = [(x,match_list.count(x)) for x in set(match_list)]
    #         print(count_tuple_list)
            count_list = [x[1] for x in count_tuple_list]
            index_max = np.where(count_list==np.max(count_list))[0][0]
            three_cate_match = count_tuple_list[index_max][0]
            three_match_list.append(three_cate_match)
        else:
            three_match_list.append(None)
    return three_match_list


# 微博舆情测试数据清洗
data_weibo = pd.read_excel(r"280_fdm_31337512.xlsx")  # weibocontent
L_text_series_weibo = data_weibo["weibocontent"].apply(clean_sent)
text_list_weibo = L_text_series_weibo.apply(tokenization)
L_index_match_weibo = match_list_top10(text_list_weibo)
L_cate_match_weibo = cate_match_top1(L_index_match_weibo)

df_weibo_test_1000 = pd.DataFrame(L_cate_match_weibo, columns=["match_result_cd"])
data_weibo = pd.concat([data_weibo, df_weibo_test_1000], axis=1)# 列合并，即行对齐,必须要保证相同的索引，如果是抽样得到的，必须要重置索引

data_weibo["三级类目"] = data_weibo["match_result_cd"].map(dict_value_key)
data_weibo.to_csv(r"weibo_test_result_lsi.csv")


# 微信舆情测试数据清洗
data_weixin = pd.read_excel(r"280_fdm_31336114.xlsx") #title+content
data_weixin["content_title"] = data_weixin["content"]+data_weixin["title"]
L_text_series_weixin = data_weixin["content_title"].apply(clean_sent)
text_list_weixin = L_text_series_weixin.apply(tokenization)

L_index_match_weixin = match_list_top10(text_list_weixin)
L_cate_match_weixin = cate_match_top1(L_index_match_weixin)

df_weixin_test_1000 =pd.DataFrame(L_cate_match_weixin, columns=["match_result_cd"])
data_weixin =pd.concat([data_weixin, df_weixin_test_1000], axis=1)# 列合并，即行对齐
data_weixin["三级类目"] = data_weixin["match_result_cd"].map(dict_value_key)
data_weixin.to_csv(r"weixin_test_result.csv")

# 新闻舆情测试数据清洗
news_data_df = pd.read_excel(r"280_fdm_31337485.xlsx")
news_data_df["key_til_cont"] = news_data_df["keyword"]+news_data_df["title"]+news_data_df["content"]

# step1:新闻舆情keyword匹配
new_ret_lists = []
for i in news_data_df['keyword']:
    value = dict_three.get(i, None)
    if value is not None:
        new_ret_lists.append(value)
    else:
        new_ret_lists.append(None)

# 保存keyword映射结果列表：[（文章索引，三级类目id,三级类目名称）,...]
dict_match_cate_three = [(j, i, dict_value_key[i]) for j, i in zip(range(len(new_ret_lists)), new_ret_lists) if i is not None]
new_data_df_keyword_match_index = [i[0] for i in dict_match_cate_three]

# 删除已经匹配过得舆情记录行
news_data_df.drop(new_data_df_keyword_match_index, inplace=True)
news_data_df.reset_index(inplace=True, drop=True)

L_text_series_news = news_data_df["key_til_cont"].apply(clean_sent)
text_list_news = L_text_series_news.apply(tokenization)
L_index_match_news = match_list_top10(text_list_news)
L_cate_match_news = cate_match_top1(L_index_match_news)

df_news_test_1000 = pd.DataFrame(L_cate_match_news, columns=["match_result_cd"])
news_data_df = pd.concat([news_data_df, df_news_test_1000], axis=1) # 列合并，即行对齐
news_data_df["三级类目"] = news_data_df["match_result_cd"].map(dict_value_key)

news_data_df.to_csv(r"news_text_result.csv")

# LDA模型建立
lda = models.ldamodel.LdaModel(corpus=doc_wine_vectors, id2word=dictionary_wine, num_topics=5, minimum_probability=0)
lda.save('test_lda.model')
lda = models.ldamodel.LdaModel.load('test_lda.model')
doc_wine_lda = lda[doc_wine_vectors]

list_doc_wine_lda = [i for i in doc_wine_lda]  # 一篇文档可能有多个主题
list_doc_origin_topic_value = []
for j in list_doc_wine_lda:
    temp = [i[1] for i in j]
    list_doc_origin_topic_value.append(temp)


def match_cos_list_top10(series):
    L_index_match = []
    for sent in series.values:
        query_bow = dictionary_wine.doc2bow(sent)
        query_lda = lda[query_bow]
        topic_vec = [i[1] for i in query_lda]  # 此处可以使用lda.get_document_topics(test_corpus)
        sim_origin = [np.dot(topic_vec, k) / (np.linalg.norm(topic_vec) * np.linalg.norm(k)) for k in list_doc_origin_topic_value]
        # 拿到top10的优质文章索引
        index_top_10 = map(sim_origin.index, heapq.nlargest(10, sim_origin))
        max_10_index = list(index_top_10)
        if [sims[i] for i in max_10_index][9] < 0.5:  # .mean(sims)
            L_index_match.append(None)
        else:
            L_index_match.append(max_10_index)
    return L_index_match


# 微博舆情
lda_L_index_match_weibo = match_cos_list_top10(text_list_weibo)  # 5分钟
lda_cate_match_weibo = cate_match_top1(lda_L_index_match_weibo)

lda_df_weibo_test_1000 = pd.DataFrame(lda_cate_match_weibo, columns=["match_result_cd_lda"])
lda_data_weibo = pd.concat([data_weibo, lda_df_weibo_test_1000], axis=1)  # 列合并，即行对齐
lda_data_weibo["三级类目_lda"] = lda_data_weibo["match_result_cd_lda"].map(dict_value_key)

lda_data_weibo.to_csv(r"lda_data_weibo.csv")

# 微信舆情
lda_L_index_match_weixin = match_cos_list_top10(text_list_weixin)
lda_cate_match_weixin = cate_match_top1(lda_L_index_match_weixin)

lda_df_weixin_test_1000 = pd.DataFrame(lda_cate_match_weixin, columns=["match_result_cd_lda"])
lda_data_weixin = pd.concat([data_weixin, lda_df_weixin_test_1000], axis=1)  # 列合并，即行对齐
lda_data_weixin["三级类目_lda"] = lda_data_weixin["match_result_cd_lda"].map(dict_value_key)

lda_data_weixin.to_csv(r"lda_data_weixin.csv")

# 新闻舆情
lda_L_index_match_news = match_cos_list_top10(text_list_news)
lda_cate_match_news = cate_match_top1(lda_L_index_match_news)

lda_df_news_test_1000 = pd.DataFrame(lda_cate_match_news, columns=["match_result_cd_lda"])
lda_data_news = pd.concat([news_data_df, lda_df_news_test_1000], axis=1)  # 列合并，即行对齐
lda_data_news["三级类目_lda"] = lda_data_news["match_result_cd_lda"].map(dict_value_key)

lda_data_news.to_csv(r"lda_data_news.csv")

