import sys
import findspark
findspark.init()
from pyspark.sql import SparkSession
import codecs
import datetime
import collections
from operator import add
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col, bround
from pyspark import Row


cate_keywords = "cate_name_relation_result1"
two_words_cate_keywords = "cate_two_words_name_relation_result1"
aspect_keywords = "aspect_knowledge.txt"  # 挂视角
tmp_path = '/tmp/'
# two_words_cate = {('男', '洁面'): ['1316', '美妆护肤', '16831', '男士面部护肤', '16840', '男士洁面'],
#                   ('男', '洗脸'): ['1316', '美妆护肤', '16831', '男士面部护肤', '16840', '男士洁面'],
#                   ('男', '洗面'): ['1316', '美妆护肤', '16831', '男士面部护肤', '16840', '男士洁面'],
#                   ('男', '乳液'): ['1316', '美妆护肤', '16831', '男士面部护肤', '16844', '男士乳液/面霜'],
#                   ('男', '面霜'): ['1316', '美妆护肤', '16831', '男士面部护肤', '16844', '男士乳液/面霜'],
#                   ('男', '面膜'): ['1316', '美妆护肤', '16831', '男士面部护肤', '16846', '男士面膜']}
two_words_default_rank = 999
title_words_default_rank = 888

if __name__ == '__main__':
    spark = SparkSession.builder.appName("userXinsheng").config("spark.some.config.option",
                                                                "some-value").enableHiveSupport().getOrCreate()
    spark.sql("set hive.exec.dynamic.partition.mode=nonstrict")
    spark.sql("set hive.exec.dynamic.partition=true")
    spark.sql("SET hive.groupby.skewindata = true")
    spark.sql("set mapred.output.compress=true")
    spark.sql("set hive.exec.compress.output=true")
    spark.sql("set mapred.output.compression.codec=com.hadoop.compression.lzo.LzopCodec")
    spark.sql("set io.compression.codecs=com.hadoop.compression.lzo.LzopCodec")
    spark.sql("set mapred.max.split.size=256000000")
    spark.sql("set mapred.min.split.size.per.node=100000000")
    spark.sql("set hive.merge.mapfiles = true")
    spark.sql("set hive.merge.mapredfiles = true")
    spark.sql("set hive.merge.size.per.task=256000000")
    spark.sql("set hive.merge.smallfiles.avgsize=16000000")
    spark.sql("use app")



    def get_value(x):
        if x is None:
            return ""
        else:
            return str(x)


    def spark_load_cate_keywords():
        '''
        load cate name keywords dict(cate name)
        such as 1319	母婴	1527	洗护用品	21420	棉柔巾	云柔巾
        :return:dict
        '''
        knowledges = collections.defaultdict(list)
        for line in codecs.open(tmp_path + cate_keywords, 'rU', 'utf-8'):
            try:
                cate1, cate1_name, cate2, cate2_name, cate3, cate3_name, word = line.strip().split('\t')
                knowledges[word] = [cate1, cate1_name, cate2, cate2_name, cate3, cate3_name]
            except Exception as ex:
                print(ex)
        return knowledges


    def spark_load_two_words_cate_keywords():
        two_words_cate = collections.defaultdict()
        for line in codecs.open(tmp_path + two_words_cate_keywords, 'rU', 'utf-8'):
            try:
                word1, word2, cate1, cate1_name, cate2, cate2_name, cate3, cate3_name = line.strip().split('\t')
                two_words_cate[(word1, word2)] = [cate1, cate1_name, cate2, cate2_name, cate3, cate3_name]
            except Exception as ex:
                print(ex)
            return two_words_cate


    def spark_load_aspect_keywords():
        '''
        load aspect keywords dict from knowledges
        such as taobao	平台	阿里系	NULL
        :return: dict
        '''
        knowledges = collections.defaultdict(list)
        for line in codecs.open(tmp_path + aspect_keywords, 'rU', 'utf-8'):
            try:
                word, aspect1, aspect2, aspect3 = line.strip().split('\t')
                knowledges[word] = [aspect1, aspect2, aspect3]
            except Exception as ex:
                print(ex)
        return knowledges


    def load_knowledge_into_dict():
        '''
        load dict func
        :return:
        '''
        knowledges = spark_load_cate_keywords()  # load cate keywords knowledge into dict
        # aspect_knowledges = spark_load_aspect_keywords()  # load aspect knowledge into dict
        two_words_cate = spark_load_two_words_cate_keywords()
        return knowledges, two_words_cate


    # cate_keywords = "cate_name_relation_result1"
    # two_words_cate_keywords = "cate_two_words_name_relation_result1"
    # aspect_keywords = "aspect_knowledge.txt"  # 挂视角
    # tmp_path = '/tmp/'


    def download_knowledge_files():
        import subprocess
        cmd = "wget -P /tmp/ http://storage.jd.local/yaung-guopeilun-dict/" + cate_keywords
        subprocess.call(cmd, shell=True)  # download cate keywords knowledge
        cmd = "wget -P /tmp/ http://storage.jd.local/yaung-guopeilun-dict/" + aspect_keywords
        subprocess.call(cmd, shell=True)  # download aspect knowledge
        cmd = "wget -P /tmp/ http://storage.jd.local/yaung-guopeilun-dict/" + two_words_cate_keywords
        subprocess.call(cmd, shell=True)  # download aspect knowledge


    def clean_keywords(keywords):
        '''
        collect_set join can not transfer to str, replace directly.
        :param keywords:
        :return:
        '''
        keywords = keywords.replace(']', '').replace('[', '').replace('\'', '')
        return keywords


    def single_keywords_from_article(keywords, result, source_text, source):
        '''
        if single article keywords in knowledge
        :param keywords: word1|freq|idf|rank, word2|freq|idf|rank
        :param result:line result
        :param source_text:knowledge dict
        :param source:knowledge
        :return:line result
        '''
        for keyword in keywords.split(','):
            if not keyword:
                continue
            word, freq, idf, rank = keyword.split('|')
            if word in source_text:
                cate1, cate1_name, \
                cate2, cate2_name, \
                cate3, cate3_name = source_text[word][0], source_text[word][1], \
                                    source_text[word][2], source_text[word][3], \
                                    source_text[word][4], source_text[word][5]
                result[(cate1, cate1_name, cate2, cate2_name, cate3, cate3_name)].append((word, int(rank), source))
        return result


    def single_keywords_from_title(keywords, result, source_text, source):
        '''
        if single article keywords in title
        :param keywords: knowledge
        :param result:line result
        :param source_text:title
        :param source: title
        :return:line result
        '''
        for word, cate in keywords.items():
            if word in source_text:
                cate1, cate1_name, \
                cate2, cate2_name, \
                cate3, cate3_name = cate[0], cate[1], \
                                    cate[2], cate[3], \
                                    cate[4], cate[5]
                result[(cate1, cate1_name,
                        cate2, cate2_name,
                        cate3, cate3_name)].append((word,
                                                    title_words_default_rank,
                                                    source))
        return result


    def two_keywords_result(keywords, result, source_text, source):
        '''
        if two article keywords in knowledge or title.two keywords dict is small so that search fast.
        :param keywords:two_words_cate
        :param result:line result
        :param source_text:article keywords str or title
        :param source:("title", "knowledge")
        :return:line result
        '''
        for words, cates in keywords.items():
            keyword_count = sum([1 for word in words if word in source_text])
            if keyword_count == 2:  # if source_text conclude both words
                cate1, cate1_name, \
                cate2, cate2_name, \
                cate3, cate3_name = cates[0], cates[1], \
                                    cates[2], cates[3], \
                                    cates[4], cates[5]
                result[(cate1, cate1_name,
                        cate2, cate2_name,
                        cate3, cate3_name)].append(('+'.join(list(words)),
                                                    two_words_default_rank,
                                                    source))
        return result


    def calculate_confidence(result, source, article_id, source_type, publish_date, dt, title):
        '''
        cates_keywords_cnt: count of cate keywords match success
        total_keywords_cnt: count of all keywords match success
        confidence = cate keyword count / total_keywords_cnt
        calculate the title and knowledge separately
        :param result: {('1316', '美妆护肤', '16831', '男士面部护肤', '16840', '男士洁面'): [('男+洁面', 999, 'knowledge')]}
        :return:
        '''
        keyword_text = collections.defaultdict(list)
        for key, value in result.items():
            keyword_text[key].extend([item[0] for item in value])
        cates_keywords_cnt = [(cate, len(keywords)) for cate, keywords in result.items()]
        total_keywords_cnt = sum([cate_keywords_cnt[1] for cate_keywords_cnt in cates_keywords_cnt])
        # cate_keywords_cnt[i]:(('1316', '美妆护肤', '16831', '男士面部护肤', '16840', '男士洁面'), 2)

        result = [(cate_keywords_cnt[0][0], cate_keywords_cnt[0][1],
                   cate_keywords_cnt[0][2], cate_keywords_cnt[0][3],
                   cate_keywords_cnt[0][4], cate_keywords_cnt[0][5],
                   cate_keywords_cnt[1], total_keywords_cnt,
                   cate_keywords_cnt[1] / total_keywords_cnt,
                   article_id, publish_date, title, source,
                   '|'.join(keyword_text[cate_keywords_cnt[0]]),
                   month_data, source_type) for cate_keywords_cnt in cates_keywords_cnt]
        return result


    def mappredit(iterator):
        '''
        map function:load knowledge;
        single_keywords_from_knowledges
        two_keywords_from_knowledges
        single_keywords_from_title
        two_keywords_from_title
        calculate confidence
        :param iterator:
        :return:
        '''
        download_knowledge_files()
        knowledges, two_words_cate = load_knowledge_into_dict()
        map_return_results = []  # map return
        for x in iterator:
            try:
                title_result, article_result = collections.defaultdict(list), collections.defaultdict(
                    list)  # line result
                article_id, source_type, publish_date, dt, \
                knowledge_aspect1, knowledge_aspect2, knowledge_aspect3, \
                title, keywords = get_value(x[0]), get_value(x[1]), get_value(x[2]), get_value(x[3]), \
                                  get_value(x[4]), get_value(x[5]), get_value(x[6]), get_value(x[7]), \
                                  get_value(x[8])
                keywords = clean_keywords(keywords)

                article_result = single_keywords_from_article(keywords=keywords,
                                                              result=article_result,
                                                              source_text=knowledges,
                                                              source='knowledge')
                keywords_w = [keyword.split('|')[0] for keyword in
                              keywords.split(',')]  # link article keywords into a str
                article_result = two_keywords_result(keywords=two_words_cate,
                                                     result=article_result,
                                                     source_text=''.join(keywords_w),
                                                     source='knowledge')

                title_result = single_keywords_from_title(keywords=knowledges,
                                                          result=title_result,
                                                          source_text=title,
                                                          source='title')

                title_result = two_keywords_result(keywords=two_words_cate,
                                                   result=title_result,
                                                   source_text=title,
                                                   source='title')

                article_confidence = calculate_confidence(article_result, 'article', article_id, source_type,
                                                          publish_date, dt, title)
                title_confidence = calculate_confidence(title_result, 'title', article_id, source_type, publish_date,
                                                        dt, title)

                map_return_results.extend(article_confidence)
                map_return_results.extend(title_confidence)

            except Exception as ex:
                raise Exception("Invalid tuple", x, ex)

        return map_return_results


    month_datas = sys.argv[2]
    for month_data in month_datas.split(','):

        today = str(datetime.date.today())
        start_month = month_data
        start_date = start_month + '-01'
        end_date = start_month + '-31'

        hql_weixin = """
            SELECT
                article_id,
                source_type,
                SUBSTR(publish_date, 1, 10) as publish_date1,

                dt,
                'NULL' as knowledge_aspect1,
                'NULL' as knowledge_aspect2,
                'NULL' as knowledge_aspect3,
                title,
                collect_list(sequence) AS keywords
            FROM
                (
                    SELECT
                        article_id,
                        source_type,
                        title,
                        publish_date,
                        --knowledge_aspect1,
                        --knowledge_aspect2,
                        --knowledge_aspect3,
                        b.dt,
                        concat_ws('|', b.word, b.freq, b.idf, b.priority) AS sequence
                    FROM
                        (
                            SELECT * FROM fdm.fdm_cis_analysis_weixinpage_chain WHERE dp = 'ACTIVE'
                        )
                        a
                    JOIN
                        (
                            SELECT
                                    source_type,
                                    article_id,
                                    word,
                                    freq,
                                    idf,
                                    priority,
                                    publish_date,
                                    knowledge_aspect1,
                                    knowledge_aspect2,
                                    knowledge_aspect3,
                                    dt
                                FROM
                                    app.app_insight_mid_new_word_optimize_da
                                WHERE
                                    dt >= '""" + start_date + """'
                                    AND dt <= '""" + today + """'
                                    AND idf != 'NULL'
                                    AND idf > 0
                                    AND source_type = '6'
                                    AND priority < 10
                        )
                        b
                    ON
                        a.mongodb_id = b.article_id
                    WHERE
                        SUBSTR(createtime, 1, 10) >= '""" + start_date + """'
                        and SUBSTR(createtime, 1, 10) <= '""" + today + """'
                        AND priority <= 10
                        AND SUBSTR(publish_date, 1, 10) >= '""" + start_date + """'
                        AND SUBSTR(publish_date, 1, 10) <= '""" + end_date + """'
                )
                zz
            GROUP BY
                article_id,
                source_type,
                SUBSTR(publish_date, 1, 10),
                dt,
                title

        """

        hql_xinwen = """
                SELECT
                    article_id,
                    source_type,
                    SUBSTR(publish_date, 1, 10) as publish_date1,
                    dt,
                    'NULL' as knowledge_aspect1,
                    'NULL' as knowledge_aspect2,
                    'NULL' as knowledge_aspect3,
                    title,
                    collect_list(sequence) AS keywords
                FROM
                    (
                        SELECT
                            article_id,
                            source_type,
                            title,
                            publish_date,
                            --knowledge_aspect1,
                            --knowledge_aspect2,
                            --knowledge_aspect3,
                            b.dt,
                            concat_ws('|', b.word, b.freq, b.idf, b.priority) AS sequence
                        FROM
                            (
                                SELECT * FROM fdm.fdm_cis_analysis_retailinsight_chain  WHERE dp = 'ACTIVE'
                            )
                            a
                        JOIN
                            (
                                SELECT
                                    source_type,
                                    article_id,
                                    word,
                                    freq,
                                    idf,
                                    priority,
                                    publish_date,
                                    knowledge_aspect1,
                                    knowledge_aspect2,
                                    knowledge_aspect3,
                                    dt
                                FROM
                                    app.app_insight_mid_new_word_optimize_da
                                WHERE
                                    dt >= '""" + start_date + """'
                                    AND dt <= '""" + today + """'
                                    AND idf != 'NULL'
                                    AND idf > 0
                                    AND source_type = '1'
                            )
                            b
                        ON
                            a.guid = b.article_id
                        WHERE
                            SUBSTR(createtime, 1, 10) >= '""" + start_date + """'
                            AND SUBSTR(createtime, 1, 10) <= '""" + today + """'
                            AND priority <= 10
                            AND SUBSTR(publish_date, 1, 10) >= '""" + start_date + """'
                            AND SUBSTR(publish_date, 1, 10) <= '""" + end_date + """'
                    )
                    zz
                GROUP BY
                    article_id,
                    source_type,
                    SUBSTR(publish_date, 1, 10),
                    dt,
                    title

            """

        hql_weibo = """
                SELECT
                    article_id,
                    source_type,
                    publish_date,
                    dt,
                    'NULL' as knowledge_aspect1,
                    'NULL' as knowledge_aspect2,
                    'NULL' as knowledge_aspect3,
                    title,
                    '' as keywords
                    --collect_list(sequence) AS keywords
                FROM
                    (
                        SELECT
                            article_id,
                            source_type,
                            publish_date,
                            --knowledge_aspect1,
                            --knowledge_aspect2,
                            --knowledge_aspect3,
                            b.dt,
                            weibocontent as title,
                            concat_ws('|', b.word, b.freq, b.idf, b.priority) AS sequence
                        FROM
                            (
                                SELECT * FROM fdm.fdm_cis_analysis_weiboinfo_chain  WHERE dp = 'ACTIVE'
                            )
                            a
                        JOIN
                            (
                                SELECT
                                    source_type,
                                    article_id,
                                    word,
                                    freq,
                                    idf,
                                    priority,
                                    publish_date,
                                    knowledge_aspect1,
                                    knowledge_aspect2,
                                    knowledge_aspect3,
                                    dt
                                FROM
                                    app.app_insight_mid_new_word_optimize_da
                                WHERE
                                    dt >= '""" + start_date + """'
                                    AND dt <= '""" + today + """'
                                    AND idf != 'NULL'
                                    AND idf > 0
                                    AND source_type = '2'
                            )
                            b
                        ON
                            a.guid = b.article_id
                        WHERE
                            SUBSTR(createtime, 1, 10) >= '""" + start_date + """'
                            AND SUBSTR(createtime, 1, 10) <= '""" + today + """'
                            AND priority <= 10
                            AND SUBSTR(publish_date, 1, 10) >= '""" + start_date + """'
                            AND SUBSTR(publish_date, 1, 10) <= '""" + end_date + """'
                    )
                    zz
                GROUP BY
                    article_id,
                    source_type,
                    publish_date,
                    dt,
                    title
        """

        if sys.argv[1] == '2':
            print(hql_weibo)
            print('正在执行微博sql。。。')
            step1 = spark.sql(hql_weibo)
        elif sys.argv[1] == '6':
            print(hql_weixin)
            print('正在执行微信sql。。。')
            step1 = spark.sql(hql_weixin)
        else:
            print(hql_xinwen)
            print('正在执行新闻sql。。。')
            step1 = spark.sql(hql_xinwen)

        total_count = step1.count()
        print("total count: ", total_count)
        print('sql执行已完成。')
        print('正在解析用户心声。。。')
        step2 = step1.rdd.mapPartitions(lambda x: mappredit(x)).cache()
        step2.toDF().show(1000)
        step2.toDF().repartition(10).createOrReplaceTempView("user_xinsheng")
        spark.sql(
            "insert overwrite table app_insight_article_label_c2m_result_month_wide_da partition(dt, source_type) select * from user_xinsheng")
        print('finished')

