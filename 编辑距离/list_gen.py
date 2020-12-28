# coding=utf-8
# version: python3.6
# 作者：Yin Pan
# 创建时间：2020/07/30 11:00
# 利用 Damerau-Levenshtein 距离的实现方式，生成品牌的变动系数
# 引用：https://web.archive.org/web/20150909134357/http://mwh.geek.nz:80/2009/04/26/python-damerau-levenshtein-distance/
# info——>list

import csv
import numpy as np
import pandas as pd
import datetime
import os
os.chdir(r"D:\gechengcheng3\Desktop\潜力类目挖掘")


def get_top_brand_list(data_dt_df, top_no):
    brand_list = []
    top_cate_dt_df = data_dt_df[data_dt_df['gmv_rank'] <= top_no].sort_values('gmv_rank')
    brand_list = top_cate_dt_df['code'].to_list()
    return brand_list


if __name__ == "__main__":  # 13:20-13:28
    file_path = r'data_info.txt'
    save_path = r'top_shop_list_org_jingxi.txt'
    df = pd.read_csv(file_path, delimiter=',').fillna("0.0000")
    print(df.dtypes)
    # 1 读取所有类目，所有月份
    all_cate_cd = set(df["item_third_cate_cd"])
    print(all_cate_cd)
    # 2 针对每个末级类目，生成每个月份的品牌排名，存储为新的DataFrame
    list_df = pd.DataFrame(columns=['cate', 'dt', 'top3', 'top5', 'top8'])
    for cate in all_cate_cd:
        cate_df = df[df['item_third_cate_cd'] == cate]
        print('正在处理:', cate)
        for dt in set(cate_df['dt']):
            cate_dt_df = cate_df[cate_df['dt'] == dt]
            list_df = list_df.append({'cate': cate,
                                      'dt': dt,
                                      'top3': str(get_top_brand_list(cate_dt_df, 3)),
                                      'top5': str(get_top_brand_list(cate_dt_df, 5)),
                                      'top8': str(get_top_brand_list(cate_dt_df, 8))
                                      }, ignore_index=True)
    # 3 输出到csv文件, lacal file
    list_df.to_csv(save_path, sep='\t', index=None)

    # # 创建日期：20202-10-10
    # # 4 输出在数据集市表中 spark转换为dataframe
    # spark_df = spark.createDataFrame(list_df)
    # # 写入表
    # spark_df.createOrReplaceTempView("df_tmp_view") # 为何要创建临时表
    # # 写入表 创建表结构
    # sbaseSQL = """
    #         create external table if not exists app_cis_c2m_brand_index_3_5_8(
    #           cate string COMMENT '三级品类ID',
    #           top3_index string COMMENT 'top3品牌',
    #           top5_index string COMMENT 'top5品牌',
    #           top8_index string COMMENT 'top8品牌')
    #         comment 'gmv top3,5,8品牌列表'
    #         partitioned by (dt string)
    #         stored as orc;
    #         """
    # spark.sql("insert overwrite table app.app_cis_c2m_brand_index_3_5_8 partition(dt = '" + dt + "') select * from df_tmp_view")
    #
