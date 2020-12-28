# coding=utf-8
# version: python3.6
# 作者：Yin Pan
# 创建时间：2020/07/30 11:00
# 利用 Damerau-Levenshtein 距离的实现方式，生成品牌的变动系数
# 引用：https://web.archive.org/web/20150909134357/http://mwh.geek.nz:80/2009/04/26/python-damerau-levenshtein-distance/
# info + list ——> index

import csv
import numpy as np
import pandas as pd
import datetime
import time
import os
os.chdir(r"D:\gechengcheng3\Desktop\潜力类目挖掘")


def damerau_levenshtein(seq1, seq2):
    """
    计算seq1,seq2的damerau距离，并输出结果矩阵
    :param seq1: 序列一，字符串 or list
    :param seq2: 序列二，字符串 or list
    :return: 编辑距离，过程矩阵
    部分参考code_snippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    """
    one_ago = None
    this_row = list(range(1, len(seq2) + 1)) + [0]
    procedure_matrix = []
    for x in range(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        two_ago, one_ago, this_row = one_ago, this_row, [0] * len(seq2) + [x + 1]
        procedure_list = []
        distance_list = []
        for y in range(len(seq2)):
            del_cost = one_ago[y] + 1  # 纵向列 + 1
            add_cost = this_row[y - 1] + 1  # 横向行 + 1
            sub_cost = one_ago[y - 1] + (seq1[x] != seq2[y])  # 字符相等时，字符不等时对角线 + 1
            cost_list = [del_cost, add_cost, sub_cost]
            cost_index = cost_list.index(min(cost_list))
            if cost_index == 0:
                procedure = ['d', [seq1[x]]]  # delete
            elif cost_index == 1:
                procedure = ['a', [seq2[y]]]  # add
            elif cost_index == 2:
                if seq1[x] != seq2[y]:
                    procedure = ['m', [seq1[x], seq2[y]]]  # 替换
                else:
                    procedure = ['m', []]
            procedure_list.append(procedure)
            this_row[y] = min(del_cost, add_cost, sub_cost)
            # This block deals with transpositions, 两者互换
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                this_row[y] = min(this_row[y], two_ago[y - 2] + 1)  ##如果
                procedure_list[-1] = ['t', [seq1[x], seq1[x - 1]]]
        procedure_matrix.append(procedure_list)
    return this_row[len(seq2) - 1], procedure_matrix


def get_solution(result_matrix):
    """
    找寻序列变化的最小编辑项
    result_matrix: damerau_levenshtein方法中形成的结果矩阵
    """
    x_index = len(result_matrix) - 1
    y_index = len(result_matrix[-1]) - 1
    change_list = []
    while x_index + y_index > 0:
        s = result_matrix[x_index][y_index]
        if s[0] == 'd':  # 删除操作，向上走一格
            x_index = x_index - 1
            change_list = change_list + s[1]
        elif s[0] == 'a':  # 增加操作，向左走一格
            y_index = y_index - 1
            change_list = change_list + s[1]
        elif s[0] == 'm':  # mismatch替换 / match不变动操作，左上走一格
            x_index = x_index - 1
            y_index = y_index - 1
            change_list = change_list + s[1]
        elif s[0] == 't':  # 交换操作，坐上走两格
            x_index = x_index - 2
            y_index = y_index - 2
            change_list = change_list + s[1]
    return change_list


def search_pct(rf_df, cate, brand, dt):
    """
    数据查询，从df中定位某品类、某品牌、某时间点的销售额占比
    :param rf_df: 数据矩阵
    :param cate: 品类id
    :param brand: 品牌id
    :param dt: dt
    :param period_m_gap: 上一周期的月份统计间隔
    :return: 该品类、品牌、dt下的销售额占比
    """
    line_sr = rf_df.loc[(rf_df['cate'] == cate) & (rf_df['dt'] == dt)
                        & (rf_df['code'] == brand)].fillna(0)
    #print(line_sr)
    #print(line_sr[['cate', 'dt', 'code', 'gmv_pct', 'gmv_pct_ly']])
    if len(line_sr) > 0:
        pct = line_sr['gmv_pct'].values[0]
        pct_ly = line_sr['gmv_pct_ly'].values[0]
    else:
        pct = 0
        pct_ly = 0
    return pct, pct_ly


def changes_calculate(list1, list2, rf_df, cate, dt):
    """
    输出序列1，序列2加权后的编辑距离
    :param list1: 序列1
    :param list2: 序列2
    :param rf_df: 数据矩阵
    :param cate: 品类id
    :param dt: dt
    :return: 加权后的编辑距离
    """
    top_list = eval(list1)  # 输入项
    top_list_ly = eval(list2)  # 输入项
    top_distance, top_matrix = damerau_levenshtein(top_list_ly, top_list)
    solution = set(get_solution(top_matrix))
    changes_sum = 0
    for s in solution:
        this_period, last_period = search_pct(rf_df, cate, s, dt)
        changes_sum = changes_sum + abs(this_period - last_period)
    return changes_sum


def get_whole_df(df, period_m_gap):
    """
    生成一个去年
    :param df: 原来的df
    :param period_m_gap:对比月份间额
    :return: whole_df 拥有当期值，与过去值的df
    """

    df['dt_m'] = pd.to_datetime(df['dt'], format='%Y-%m')
    df_current = df.copy()
    df_past = df.copy()
    df_past['dt_m'] = df_past['dt_m'].apply(lambda x: x + pd.DateOffset(months=period_m_gap))
    df_whole = pd.merge(df_current, df_past, how='inner', on=['dt_m', 'cate'], suffixes=('', '_ly'))
    return df_whole


def get_whole_rf_df(df, period_m_gap):
    """
    生成一个去年
    :param df: 原来的df
    :param period_m_gap:对比月份间额
    :return: whole_df 拥有当期值，与过去值的df
    """
    df['dt_m'] = pd.to_datetime(df['dt'], format='%Y-%m')
    df_current = df.copy()
    df_past = df.copy()
    df_past['dt_m'] = df_past['dt_m'].apply(lambda x: x + pd.DateOffset(months=period_m_gap))
    df_whole = pd.merge(df_current, df_past, how='outer', on=['dt_m', 'cate', 'code'], suffixes=('', '_ly'))
    return df_whole


if __name__ == "__main__":
    file_path = r'top_shop_list_org_jingxi.txt'
    refer_path = r'top_shop_info_org.txt'
    save_path = r'top_shop_change_index_org_jingxi.txt'
    list_df = pd.read_csv(file_path, delimiter='\t')
    rf_df = pd.read_csv(refer_path, delimiter='\t')
    print(rf_df.dtypes)
    rf_df = rf_df.rename(columns={'item_third_cate_cd': 'cate'})
    period_month_gap = 12
    # 开始构造merge矩阵
    whole_list_df = get_whole_df(list_df, period_month_gap)
    # temp_df = get_whole_df(list_df, period_month_gap)[1]
    print('list矩阵构造完毕')
    # print(whole_list_df.dtypes)
    all_cate_cd = set(whole_list_df["cate"])
    result_df = pd.DataFrame(columns=['cate', 'dt', 'top3_index', 'top5_index', 'top8_index'])
    # 按照品类计算，加速效率
    for cate in all_cate_cd:
        t1 = time.time()
        cate_df = whole_list_df[whole_list_df['cate'] == cate]
        print('正在处理:', cate)
        cate_rf_df = rf_df[rf_df['cate'] == cate]
        cate_whole_rf_df = get_whole_rf_df(cate_rf_df, period_month_gap)
        # print(cate_whole_rf_df.dtypes)
        for index, row in cate_df.iterrows():
            dt = row['dt']
            if len(eval(row['top3'])) == 0:
                top3_change_index, top5_change_index, top8_change_index = 0, 0, 0
            else:
                top3_change_index = changes_calculate(row['top3_ly'], row['top3'], cate_whole_rf_df, cate, dt)
                top5_change_index = changes_calculate(row['top5_ly'], row['top5'], cate_whole_rf_df, cate, dt)
                top8_change_index = changes_calculate(row['top8_ly'], row['top8'], cate_whole_rf_df, cate, dt)
            result_df = result_df.append({'cate': cate,
                                          'dt': dt,
                                          'top3_index': top3_change_index,
                                          'top5_index': top5_change_index,
                                          'top8_index': top8_change_index
                                          }, ignore_index=True)
            print('处理完毕:', cate, dt, top3_change_index, top5_change_index, top8_change_index)
        t2 = time.time()
        print('处理时间',t2-t1)
    result_df.to_csv(save_path, sep='\t')