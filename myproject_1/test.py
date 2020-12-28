from snownlp import SnowNLP
text = '你站在桥上看风景，看风景的人在楼上看你。明月装饰了你的窗子，你装饰了别人的梦'

s = SnowNLP(text)

# 分词
print(s.words)

text2 = [["不是"], ["不好"], ["不是不好"], ["很好"], ["不是很好看"]]
for j in text2:
    ss = SnowNLP(j[0])
    print(j, ss.sentiments)

#分词
text3 = '李达康就是这样的人，她穷哭出声，不攀龙附凤，不结党营私，不同流合污，不贪污受贿，也不伪造政绩，手下贪污出事了他自责用人不当，服装厂出事了他没想过隐瞒，後面這些是繁體字'
s = SnowNLP(text3)
print(s.words)
#词性标注
tags = [x for x in s.tags]
print(tags)
#断句
print(s.sentences)

#情感分析 Jiagu也支持情感分析，可以对比一下二者的效果
text_1 = '这部电影真心棒，全程无尿点'
text_2 = '这部电影简直烂到爆'
s1 = SnowNLP(text_1)
s2 = SnowNLP(text_2)
print(text_1, s1.sentiments)
print(text_2, s2.sentiments)

# 拼音
print(s.pinyin)

# 繁体转简体
print(s.han)

# 关键字抽取
text4 = '''
北京故宫 是 中国 明清两代 的 皇家 宫殿 ， 旧 称为 紫禁城 ， 位于 北京 中轴线 的 中心 ， 是 中国 古代 宫廷 建筑 之 精华 。 北京故宫 以 三 大殿 为 中心 ， 占地面积 72 万平方米 ， 建筑面积 约 15 万平方米 ， 有 大小 宫殿 七十 多座 ， 房屋 九千余 间 。 是 世界 上 现存 规模 最大 、 保存 最为 完整 的 木质 结构 古建筑 之一 。 
北京故宫 于 明成祖 永乐 四年 （ 1406 年 ） 开始 建设 ， 以 南京 故宫 为 蓝本 营建 ， 到 永乐 十八年 （ 1420 年 ） 建成 。 它 是 一座 长方形 城池 ， 南北 长 961 米 ， 东西 宽 753 米 ， 四面 围有 高 10 米 的 城墙 ， 城外 有 宽 52 米 的 护城河 。 紫禁城 内 的 建筑 分为 外朝 和内廷 两 部分 。 外朝 的 中心 为 太和殿 、 中和殿 、 保和殿 ， 统称 三 大殿 ， 是 国家 举行 大 典礼 的 地方 。 内廷 的 中心 是 乾清宫 、 交泰 殿 、 坤宁宫 ， 统称 后 三宫 ， 是 皇帝 和 皇后 居住 的 正宫 。   [ 1 ]   
北京故宫 被誉为 世界 五大 宫之首 （ 法国 凡尔赛宫 、 英国 白金汉宫 、 美国白宫 、 俄罗斯 克里姆林宫 ） ， 是 国家 AAAAA 级 旅游 景区 ，   [ 2 - 3 ]     1961 年 被 列为 第一批 全国 重点 文物保护 单位 ；   [ 4 ]     1987 年 被 列为 世界 文化遗产 。   [ 5 ]   
2012 年 1 月 至 2018 年 6 月 ， 故宫 累计 接待 观众 达到 1 亿人次 。 2019 年 起 ， 故宫 将 试行 分 时段 售票   [ 6 ]     。 2018 年 9 月 3 日 ， 故宫 养心殿 正式 进入 古建筑 研究性 保护 修缮 工作 的 实施 阶段 。   [ 7 ]     2019 年 3 月 4 日 ， 故宫 公布 了 2019 年 下半年 展览 计划 。   [ 8 ]   
'''

s4 = SnowNLP(text4)
print(s4.keywords(limit=10)) # ['故宫', '年', '米', '外', '中心', '世界', '建筑', '北京', '宫', '保护']

# 八、概括总结文章
print(s4.summary(limit=4))


# 九、信息衡量
'''
TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。

TF词频越大越重要，但是文中会的“的”，“你”等无意义词频很大，却信息量几乎为0，这种情况导致单纯看词频评价词语重要性是不准确的。因此加入了idf

IDF的主要思想是：如果包含词条t的文档越少，也就是n越小，IDF越大，则说明词条t越重要

TF-IDF综合起来，才能准确的综合的评价一词对文本的重要性。
'''
s5 = SnowNLP([
    ['性格', '善良'],
    ['温柔', '善良', '善良'],
    ['温柔', '善良'],
    ['好人'],
    ['性格', '善良'],
])
print(s5.tf)  # [{'性格': 1, '善良': 1}, {'温柔': 1, '善良': 2}, {'温柔': 1, '善良': 1}, {'好人': 1}, {'性格': 1, '善良': 1}]
print(s5.idf)  # {'性格': 0.33647223662121295, '善良': -1.0986122886681098, '温柔': 0.33647223662121295, '好人': 1.0986122886681098}


# 十、文本相似性
print(s5.sim(['温柔']))  # [0, 0.2746712135683371, 0.33647223662121295, 0, 0] ；维度为5，计算的是和每一篇文章得相似性，此处一共有五篇文章
print(s5.sim(['善良']))  # [-1.0986122886681098, -1.3521382014376737, -1.0986122886681098, 0, -1.0986122886681098]
print(s5.sim(['好人']))  # [0, 0, 0, 1.4175642434427222, 0]
print(s5.sim(["真善美"]))  # [0, 0, 0, 0, 0] 训练集中如果没有想匹配的词，则输出结果均为零


"""
NLTK分词测试
"""
import nltk
from nltk.corpus import stopwords
from nltk.corpus import brown  # brown语料库只是nltk中的一种语料库，还有其他很多种：nps_shat,conll2000,treebank等等
import numpy as np

# 分词
text0 = "Sentiment analysis is a challenging subject in machine learning.\
 People express their emotions in language that is often obscured by sarcasm,\
  ambiguity, and plays on words, all of which could be very misleading for \
  both humans and computers.".lower()  # string.lower()
text_list = nltk.word_tokenize(text0)  # 43
# 去掉标点符号
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
text_list = [word for word in text_list if word not in english_punctuations]  # 38
# 去掉停用词
stops = set(stopwords.words("english"))  # 179个停用词
text_list = [word for word in text_list if word not in stops]  # 20

# 词性标注
word_tags = nltk.pos_tag(text_list)
"""
 ('people', 'NNS'),名词复数
 ('often', 'RB'),副词
"""
brown_taged = nltk.corpus.brown.tagged_words()
# brown_taged_simplify = nltk.corpus.brown.tagged_words(simplify_tags=True) python2 支持此种写法，Python3 已经不支持
"""
NLTK中包括的若干语料库已经标注了词性,1161192个已经标注词性，brown_taged是一个列表，长度1161192,其内的元素是元组——(词,词性)
"""

# 自动标注
brown_tagged_sents = brown.tagged_sents(categories='news')  # 是一个列表，列表元素是元组（词，词性）
brown_sents = brown.sents(categories='news')  # 是一个列表，列表元素是词，有4623篇语料，即大列表里面有4623个小列表
# 默认标注
tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
tag_fd = nltk.FreqDist(tags)
print(tag_fd.max())
tag_fd.plot(cumulative=True)

raw = 'I do not like green eggs and ham, I do not like them Sam I am!'
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('IN')
print(default_tagger.tag(tokens))
print(default_tagger.evaluate(brown_tagged_sents))  # 自动标注好的，比较强制标注为IN的，计算标注准确率 = 标注正确/总标注数
"""
[('I', 'NN'), ('do', 'NN'), ('not', 'NN'), ('like', 'NN'), ('green', 'NN'), ('eggs', 'NN'), ('and', 'NN'), ('ham', 'NN'), (',', 'NN'), ('I', 'NN'), ('do', 'NN'), ('not', 'NN'), ('like', 'NN'), ('them', 'NN'), ('Sam', 'NN'), ('I', 'NN'), ('am', 'NN'), ('!', 'NN')]
0.13089484257215028
"""

# 正则表达式标注器
patterns = [(r'.*ing$', 'VBG'), (r'.*ed$', 'VBD'), (r'.*es$', 'VBZ'), (r'.*ould$',  'MD'),\
            (r'.*\'s$', 'NN$'), (r'.*s$', 'NNS'), (r'^-?[0-9]+(.[0-9]+)?$', 'CD'), (r'.*', 'NN')]
regexp_tagger = nltk.RegexpTagger(patterns)
print(regexp_tagger.tag(brown_sents[3]))
print(regexp_tagger.evaluate(brown_tagged_sents))
"""
[('``', 'NN'), ('Only', 'NN'), ('a', 'NN'), ('relative', 'NN'), ('handful', 'NN'), ('of', 'NN'), ('such', 'NN'), ('reports', 'NNS'), ('was', 'NNS'), ('received', 'VBD'), ("''", 'NN'), (',', 'NN'), ('the', 'NN'), ('jury', 'NN'), ('said', 'NN'), (',', 'NN'), ('``', 'NN'), ('considering', 'VBG'), ('the', 'NN'), ('widespread', 'NN'), ('interest', 'NN'), ('in', 'NN'), ('the', 'NN'), ('election', 'NN'), (',', 'NN'), ('the', 'NN'), ('number', 'NN'), ('of', 'NN'), ('voters', 'NNS'), ('and', 'NN'), ('the', 'NN'), ('size', 'NN'), ('of', 'NN'), ('this', 'NNS'), ('city', 'NN'), ("''", 'NN'), ('.', 'NN')]
0.20326391789486245 标注正确率
"""

# 查询标注器：找出100个最频繁的词，存储它们最有可能的标记。然后可以使用这个信息作为
# "查询标注器"（NLTK UnigramTagger）的模型
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = list(fd.keys())[:100]
likely_tags = dict((word, cfd[word].max()) for word in most_freq_words)
# baseline_tagger = nltk.UnigramTagger(model=likely_tags)
# 许多词都被分配了None标签，因为它们不在100个最频繁的词中，可以使用backoff参数设置这些词的默认词性
baseline_tagger = nltk.UnigramTagger(model=likely_tags, backoff=nltk.DefaultTagger('NN'))
print(baseline_tagger.evaluate(brown_tagged_sents))





word_tag_pairs = nltk.bigrams(brown_taged)
list_pairs = list(nltk.FreqDist(a[1] for (a, b) in word_tag_pairs if b[1] == "N"))
# for (a, b) in word_tag_pairs[:5]:
#     if b[1] == "N":
#         print(a[1])




# 清华分词器
import thulac

thu1 = thulac.thulac()  #默认模式
text = thu1.cut("我爱北京天安门", text=True)  #进行一句话分词
print(text)

# 中科院分词器
# 暂无实例

# 哈工大分词器
from ltp import LTP
ltp = LTP()

# baidu 分词器
from LAC import LAC

# 装载分词模型
lac1 = LAC()

# 单个样本输入，输入为Unicode编码的字符串
text = u"LAC是个优秀的分词工具"
seg_result = lac.run(text)

# 批量样本输入, 输入为多个句子组成的list，平均速率会更快
texts = [u"LAC是个优秀的分词工具", u"百度是一家高科技公司"]
seg_result = lac.run(texts)

""""""""""""""""""""""""""""""""""""""""测试对比"""""""""""""""""""""""""""""""""

"in gub shn nxe".split(" ")