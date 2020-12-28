4.1 熟悉PEP-8，参考百度Python编码规范，改写步骤3中的代码
# -*- coding: utf-8 -*-
#step1:文本清洗
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

df_nan_stats = []
for dt in set(raw_data["dt_month"]):
    temp_table_name = "raw_data_"+dt
    temp_table_name = raw_data.loc[raw_data['dt_month'] == dt, :]
    temp_table_name["nan_rate"] = temp_table_name["search_time"]/sum(temp_table_name.loc[:, "search_time"])
    temp_table_name.reset_index(drop=True, inplace=True)
    df_nan_stats.append(1-temp_table_name['nan_rate'][0])
df_nan_stats = pd.DataFrame(df_nan_stats, columns=["keyword_notnull_占比"], index=set(raw_data["dt_month"])).sort_index()

#统计总的搜索次数和非空搜索占比
df_search_time_all_stats = []
for dt in set(raw_data["dt_month"]):
    temp_table_name = "raw_data_"+dt
    temp_table_name=raw_data.loc[raw_data['dt_month'] == dt, :]
    temp = sum(temp_table_name.search_time)
    df_search_time_all_stats.append(temp)
df_search_time_all_stats=pd.DataFrame(df_search_time_all_stats, columns=["搜索次数"], index=set(raw_data["dt_month"])).sort_index()
df_search_time_all_stats['keyword_notnull_占比']=df_nan_stats['keyword_notnull_占比'].copy()
df_stats = df_search_time_all_stats.copy()
df_search_time_all_stats.drop("keyword_notnull_占比", inplace=True, axis=1)
print(df_stats)

#清洗文本并保存在corpus_clean.txt中
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
f = open("corpus_clean.txt", "r")   #设置文件对象
string = f.readlines()     #将txt文件的所有内容读入到字符串string中
f.close()
string=string[12:].copy() #去除nan
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