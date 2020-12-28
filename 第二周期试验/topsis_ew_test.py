# coding=utf-8
# version: python3.6
# 作者：Yin Pan
# 创建时间：2020/09/17 11:00
# 潜力挖掘项目中，利用清洗好的底层数据，构建项目模型，并且输出最终结果
# 结构方程包参考资料：https://github.com/GoogleCloudPlatform/plspm-python；https://plspm.readthedocs.io/en/latest/
# all
from sklearn.metrics import mean_squared_error
import numpy as np
import argparse
from sklearn.externals import joblib

import pandas as pd
import plspm.config as c
from plspm.plspm import Plspm
from plspm.scale import Scale
from plspm.scheme import Scheme
from plspm.mode import Mode
import csv
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def score_path_matrix():
    structure = c.Structure()
    structure.add_path(["User", "UserRate", "Traffic", "TrafficRate", "Sale", "SaleRate", "CR", "TopSKUBrand", \
                        "CRRate", "BrandChange", "Season"], ["CategoryPotential"])
    return structure.path()


def score_add_lv(config):
    config.add_lv("User", Mode.A, c.MV("order_user"), c.MV("search_user"))
    config.add_lv("UserRate", Mode.A, c.MV("order_user_rate"), c.MV("search_user_rate"))
    config.add_lv("Traffic", Mode.A, c.MV("pv"), c.MV("uv"), c.MV("search"))
    config.add_lv("TrafficRate", Mode.A, c.MV("pv_rate"), c.MV("uv_rate"), c.MV("search_rate"))
    config.add_lv("Sale", Mode.A, c.MV("sale"), c.MV("gmv"))
    config.add_lv("SaleRate", Mode.A, c.MV("sale_rate"), c.MV("gmv_rate"))
    config.add_lv("CR", Mode.A, c.MV("cr3"), c.MV("cr5"), c.MV("cr8"))
    config.add_lv("TopSKUBrand", Mode.A, c.MV("top10brand"), c.MV("top20brand"), c.MV("top50brand"))
    config.add_lv("CRRate", Mode.A, c.MV("cr3_rate"), c.MV("cr5_rate"), c.MV("cr8_rate"))
    config.add_lv("BrandChange", Mode.A, c.MV("brand_change3"), c.MV("brand_change5"), c.MV("brand_change8"))
    config.add_lv("Season", Mode.B, c.MV("month", Scale.NOM))  # 数据集较小，时间序列方法不适用，目前仅观察一级品类是否有特殊月份起伏
    config.add_lv("CategoryPotential", Mode.B, c.MV("growth_pct"), c.MV("adjust_rate"))  # Mode.B 进行formative测量


def fit_model(X_df, Y_df):
    this_model = sm.OLS(Y_df, X_df).fit()
    return this_model


def get_vif(X):
    vif_df = pd.DataFrame()
    vif_df["variables"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_df


chinese = ['item_first_cate_cd', 'item_first_cate_name', 'item_second_cate_cd', 'item_second_cate_name', \
           'item_third_cate_cd', 'item_third_cate_name', 'item_fourth_cate_cd', 'item_fourth_cate_name',
           'item_last_cate_cd', \
           'item_last_cate_name', 'dt', \
           "e_CategoryPotential", "用户综合得分", "流量综合得分", "销售综合得分", "竞争力综合得分", \
           "用户占比得分", "用户同比得分", "流量占比得分", "流量同比得分", \
           "销售占比得分", "销售同比得分", "集中度占比得分", "集中度同比得分", "topsku店铺数得分", "店铺变动得分"]

origin = ['item_first_cate_cd', 'item_first_cate_name', 'item_second_cate_cd', 'item_second_cate_name', \
          'item_third_cate_cd', 'item_third_cate_name', 'item_fourth_cate_cd', 'item_fourth_cate_name',
          'item_last_cate_cd', \
          'item_last_cate_name', 'dt', \
          'e_CategoryPotential', \
          'UserScore', 'TrafficScore', 'SaleScore', 'CompeteScore', \
          'User', 'UserRate', 'Traffic', 'TrafficRate', 'Sale', 'SaleRate', \
          'CR', 'CRRate', 'TopSKUBrand', 'BrandChange']


def outer_deal(df_cate, data_final_float_type):
    """
    异常值处理
    """
    for i in data_final_float_type:
        Q1 = df_cate[i].quantile(q=0.25)
        Q3 = df_cate[i].quantile(q=0.75)
        IQR = Q3 - Q1
        check1 = any(df_cate[i] > Q3 + 1.5 * IQR)
        if check1 == True:
            print('{0}特征有高于上限的异常值'.format(i))
            alter_index = list(df_cate.loc[df_cate[i] > Q3 + 1.5 * IQR].index)
            df_cate.loc[alter_index, i] = Q3 + 1.5 * IQR
            check1 = any(df_cate[i] > Q3 + 1.5 * IQR)
            print(check1)
            if check1 == False:
                print('{0}特征异常值处理完毕\n'.format(i))
        else:
            print('{0}特征无高于上限的异常值\n'.format(i))

    for i in data_final_float_type:
        Q1 = df_cate[i].quantile(q=0.25)
        Q3 = df_cate[i].quantile(q=0.75)
        IQR = Q3 - Q1
        check2 = any(df_cate[i] < Q1 - 1.5 * IQR)
        if check2 == True:
            print('{0}特征有低于下限的异常值'.format(i))
            alter_index = list(df_cate.loc[df_cate[i] < Q1 - 1.5 * IQR].index)
            df_cate.loc[alter_index, i] = Q1 - 1.5 * IQR
            check2 = any(df_cate[i] < Q1 - 1.5 * IQR)
            if check2 == False:
                print('{0}特征异常值处理完毕\n'.format(i))
        else:
            print('{0}特征无低于下限异常值\n'.format(i))

    print('异常值检测并处理完毕\n')
    df = df_cate.copy()
    return df


transform_df = ['item_first_cate_cd', 'item_first_cate_name', 'item_second_cate_cd', \
                'item_second_cate_name', 'item_third_cate_cd', 'item_third_cate_name', \
                'item_fourth_cate_cd', 'item_fourth_cate_name', 'item_last_cate_cd', \
                'item_last_cate_name', 'product_id', 'dt', \
                'uv_rate', 'pv_rate', 'search_rate', 'sale_rate', 'gmv_rate', 'search_user_rate', 'order_user_rate', \
                'uv', 'pv', 'search', 'sale', 'gmv', 'search_user', 'order_user', \
                'cr3', 'cr5', 'cr8', 'cr3_rate', 'cr5_rate', 'cr8_rate', \
                'brand_change3', 'brand_change5', 'brand_change8', \
                'top10brand', 'top20brand', 'top50brand', \
                'growth_pct', 'adjust_rate', 'month'
                ]

origin_df = ['item_first_cate_cd', 'item_first_cate_name', 'item_second_cate_cd', \
             'item_second_cate_name', 'item_third_cate_cd', 'item_third_cate_name', \
             'item_fourth_cate_cd', 'item_fourth_cate_name', 'item_last_cate_cd', \
             'item_last_cate_name', 'product_id', 'dt', \
             'uv_rate', 'pv_rate', 'search_rate', 'sale_cnt_rate', 'gmv_rate', 'user_rate', 'order_rate', \
             'uv_pct', 'pv_pct', 'search_pct', 'sale_cnt_pct', 'gmv_pct', 'user_pct', 'order_pct', \
             'cr3', 'cr5', 'cr8', 'cr3_rate', 'cr5_rate', 'cr8_rate', \
             'top3_index', 'top5_index', 'top8_index', \
             'top10_brand_nums', 'top20_brand_nums', 'top50_brand_nums', \
             'growth_pct', 'adjust_rate', 'months']


def drop_duplicate(df_cate, x):
    """
    重复值处理，去除每一个特征中重复值高于百分之95的特征
    """

    if ((df_cate[x].value_counts().sort_values(ascending=False).max() / df_cate[x].value_counts().sum()) > 0.95):
        print('不适宜拟合模型的一级类目为{0}，该特征内重复值占比为{1:.4f}\n'.format(x, (
                df_cate[x].value_counts().sort_values(ascending=False).max() / df_cate[x].value_counts().sum())))
        del df_cate[x]

import math
import warnings

warnings.filterwarnings("ignore")


# 定义熵值法函数
def cal_weight(x):
    '''熵值法计算变量的权重'''
    # 标准化
    x = x.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))
    #     print(x)
    #     print("归一化之后的最大值为{}.".format(x.max()))
    #     print("归一化之后的最小值为{}.".format(x.min()))

    '''
    如果数据实际不为零，则赋予最小值
    if x==0:
        x=0.00001
    else:
        pass
    '''
    # 求k
    rows = x.index.size  # 行
    cols = x.columns.size  # 列
    k = 1.0 / math.log(rows)

    lnf = [[None] * cols for i in range(rows)]

    # 矩阵计算--
    # 信息熵

    x = np.array(x)
    lnf = [[None] * cols for i in range(rows)]
    lnf = np.array(lnf)
    for i in range(0, rows):
        for j in range(0, cols):
            if x[i][j] == 0:
                lnfij = 0.0
            else:
                p = x[i][j] / x.sum(axis=0)[j]
                lnfij = math.log(p) * p * (-k)
            lnf[i][j] = lnfij
    lnf = pd.DataFrame(lnf)
    E = lnf

    # 计算冗余度
    d = 1 - E.sum(axis=0)
    # 计算各指标的权重
    w = [[None] * 1 for i in range(cols)]
    for j in range(0, cols):
        wj = d[j] / sum(d)
        w[j] = wj
        # 计算各样本的综合得分,用最原始的数据

    w = pd.DataFrame(w)
    return w

import os
os.chdir(r"D:\gechengcheng3\Desktop\潜力类目挖掘")

if __name__ == "__main__":
    #     check_path = r"outputs_results_25/check_outputs"
    file_path = r'last_7_test_data_v2.txt'
    saving_path = r'./test_1211/offline_1topsis_1zs_ew_half'
    # saving_data_test = r'./outputs_results_7_1207/test_data'
    dict_df = dict(zip(origin_df, transform_df))
    df = pd.read_csv(file_path, delimiter='\t', index_col=['dt', 'item_last_cate_cd'], encoding="utf8")
    df.rename(columns=dict_df, inplace=True)
    df.drop(columns=["product_id"], inplace=True)

    # 预处理step1检测并去除重复行
    for i in np.array(df.columns[5:]):
        drop_duplicate(df, i)
    print('数据集是否存在重复观测: ', any(df.duplicated()))
    print('\n重复值处理完毕\n')

    # 标准化数据参与潜变量计算和最终的模型拟合 来自业务输入
    with open(saving_path + r'/0_0模型解释度_R方.csv', "w", newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(('cate', 'dt', 'R_2'))

    # weight_save
    with open(saving_path + r'/0_1weight_save.csv', "w", newline='', encoding='GBK') as f:
        writer = csv.writer(f)
        writer.writerow(('cate', 'growth_pct', 'adjust_rate'))

    # 读取所有一级类目列表 每一个一级类目拟合一个模型
    # cate_list = list(set(df["item_first_cate_name"]))
    cate_list = list(set(df["item_first_cate_name"]))
    # 按照一级类目进行循环

    #     cates = []  # 报错类目回溯矩阵记录
    cates_features = []
    for cate in cate_list:  # 每一个一级类目拟合一个模型
        #         try:
        cate_saving_path = saving_path + r'/' + cate.replace('/', '')
        #         check_saving_path = check_path + r'/' + cate.replace('/', '')
        cate_df = df[df["item_first_cate_name"] == cate].copy()
        kongzhi = max([cate_df[col].isnull().sum() for col in cate_df.columns[9:]])  # 仅考虑数值型变量的缺失
        if len(cate_df) - kongzhi >= 90 and kongzhi / len(cate_df) <= 0.36 and len(
                cate_df.dropna(axis=0, how='any')) >= 90:  # 单个特征的维度和完整记录数维度
            print("一级类目 {} 数据满足建模条件".format(cate))

            # step2每一个类目单独处理异常值，因为是单独建模，必须单独处理
            data_final_float_type = list(cate_df.dtypes.loc[cate_df.dtypes.values == 'float64'].index)
            cate_df = outer_deal(cate_df, data_final_float_type)

            cate_df_filled = cate_df.fillna(cate_df.mean())  # 利用均值填充缺失值(可能存在一定风险)进行潜变量计算,数据的时间跨度越宽风险越小 否则风险越大

            # 定义路径和度量模型
            config = c.Config(score_path_matrix(), default_scale=Scale.NUM)
            score_add_lv(config)
            # 进行潜变量计算 所有潜变量均为均值为0标准差为1的变量
            plspm_calc = Plspm(cate_df_filled, config, Scheme.PATH, iterations=1000, tolerance=1e-06, bootstrap=False)
            scores = plspm_calc.scores()
            scores.drop_duplicates(keep="first", inplace=True)
            scores_copy = scores.copy()

            # 全量归一化
            # scores_zs = scores.loc[:, "Season":].apply(lambda x: (x - min(x)) / (max(x) - min(x)))
            # scores_copy.drop(['Season', 'BrandChange', 'CRRate', 'TopSKUBrand', 'CR', 'SaleRate',
            #                   'Sale', 'TrafficRate', 'Traffic', 'UserRate', 'User',
            #                   'CategoryPotential'], axis=1, inplace=True)
            # scores = pd.concat([scores_copy, scores_zs], axis=1)

            # 部分归一化
            scores_zs = scores.loc[:, ["CR", "TopSKUBrand", "BrandChange"]].apply(lambda x: (x - min(x)) / (max(x) - min(x)))
            scores_copy.drop(["CR", "TopSKUBrand", "BrandChange"], axis=1, inplace=True)
            scores = pd.concat([scores_copy, scores_zs], axis=1)

            # 输出潜变量指标
            # plspm_calc.inner_summary().to_csv(cate_saving_path + '_1_1潜变量内模型指标.csv', encoding='GBK')
            # plspm_calc.unidimensionality().to_csv(cate_saving_path + '_1_2潜变量外模型指标.csv', encoding='GBK') # only meaningful for reflective / mode A blocks

            # 潜变量直接关系系数 潜变量和显变量关系系数
            plspm_calc.inner_model().to_csv(cate_saving_path + '_1_3潜变量内模型.csv', encoding='GBK')
            plspm_calc.outer_model().to_csv(cate_saving_path + '_1_4潜变量外模型.csv', encoding='GBK')
            # scores.corr().to_csv(cate_saving_path + '_1_5潜变量相关系数.csv', encoding='GBK')  # 指标区分性
            # 潜变量得分 与 原始矩阵合并
            cate_df_final = pd.merge(cate_df, scores, how='left', left_index=True, right_index=True)
                                     # suffixes=('', '_latent'))

            # 熵权法赋权计算潜力值 Y
            df_y = cate_df_final.loc[:, "growth_pct":"adjust_rate"].copy()
            print("处理前{}条".format(len(df_y)))
            df_y.dropna(inplace=True, axis=0, how="any")

            print("处理后{}条".format(len(df_y)))

            # topsis得分计算
            scores_dict = {}
            for col in df_y.columns:
                Q1 = df_y[col].quantile(q=0.25)
                Q3 = df_y[col].quantile(q=0.75)
                IQR = Q3 - Q1
                good_score = Q3 + 1.5 * IQR
                bad_score = Q1 - 1.5 * IQR
                scores_dict[col] = (good_score, bad_score)

            df_y["growth_pct_score"] = np.abs(df_y["growth_pct"] - scores_dict["growth_pct"][0]) / (
                            np.abs(df_y["growth_pct"] - scores_dict["growth_pct"][0]) + np.abs(
                        df_y["growth_pct"] - scores_dict["growth_pct"][1]))
            df_y["adjust_rate_score"] = np.abs(df_y["adjust_rate"] - scores_dict["adjust_rate"][0]) / (
                            np.abs(df_y["adjust_rate"] - scores_dict["adjust_rate"][0]) + np.abs(
                        df_y["adjust_rate"] - scores_dict["adjust_rate"][1]))

            w = cal_weight(df_y.iloc[:, :2])  # 调用cal_weight
            w.index = df_y.columns[:2]
            w.columns = ['weight']
            w_gp, w_ar = list(w["weight"].values)[0], list(w["weight"].values)[1]
            df_y["potential_Y"] = w_gp * df_y["growth_pct_score"] + w_ar * df_y["adjust_rate_score"]

            df_y_copy = df_y.copy()

            # 计算均值和标准差
            temp_mean = np.mean(df_y.loc[:, "potential_Y"])
            temp_std = np.std(df_y.loc[:, "potential_Y"])
            # df_y_zs = df_y.loc[:, "potential_Y"].apply(lambda x: (-1*(x - temp_mean) / temp_std)+1)
            df_y_zs = df_y.loc[:, "potential_Y"].apply(lambda x: ((x - temp_mean) / temp_std))

            df_y_copy.drop(["potential_Y", "growth_pct_score", "adjust_rate_score", "adjust_rate", 'growth_pct'], axis=1, inplace=True)
            df_y = pd.concat([df_y_copy, df_y_zs], axis=1)
            df_y.rename(columns={"potential_Y": "CategoryPotential"}, inplace=True)
            # cate_df_final.drop_duplicates(keep="first", inplace=True)
            cate_df_final = pd.merge(cate_df_final, df_y, how='left', left_index=True, right_index=True,
                                     suffixes=('', '_score'))
            cate_df_final_save = cate_df_final.copy()
            cate_df_final_save.to_csv(cate_saving_path + r"_test_data.csv", index=False, header=True)
            cate_df_final.drop("CategoryPotential", axis=1, inplace=True)
            cate_df_final.rename(columns={"CategoryPotential_score": "CategoryPotential"}, inplace=True)

            with open(saving_path + r'/0_1weight_save.csv', "a", newline='', encoding='GBK') as f:
                writer = csv.writer(f)
                writer.writerow((cate, w_gp, w_ar))

            cate_df_final.drop_duplicates(keep="first", inplace=True)
            # 准备数据进行多元回归分析
            # 生成调节变量
            mod_list = [('User', 'UserRate'), ('Traffic', 'TrafficRate'), ('Sale', 'SaleRate'), ('CR', 'SaleRate'),
                        ('CR', 'CRRate'), ('Sale', 'TopSKUBrand'), ('Sale', 'BrandChange')]
            for mod in mod_list:
                cate_df_final[str(mod[0] + '_' + mod[1])] = cate_df_final[mod[0]] * cate_df_final[mod[1]]
            # 引入二次项
            cate_df_final["CR_2"] = cate_df_final["CR"] * (1 - cate_df_final["CR"])
            cate_df_final["TopSKUBrand_2"] = cate_df_final["TopSKUBrand"] * (1 - cate_df_final["TopSKUBrand"])
            cate_df_final["BrandChange_2"] = cate_df_final["BrandChange"] * (1 - cate_df_final["BrandChange"])

            cate_df_train = cate_df_final.dropna(axis=0, how='any')  # 仅利用非空行进行统计
            print(cate_df_train.shape[0], cate_df_final.shape[0])
            if cate_df_train.shape[0] < 90:  # 完整记录的维度 或者多个特征的维度
                print("类目 {} 建模失败".format(cate))
                cates_features.append(cate)
                continue

            # 构建训练数据
            Y = cate_df_train['CategoryPotential']
            # X = cate_df_train.loc[:, 'Season':].drop(labels='CategoryPotential', axis=1)
            X = cate_df_train.loc[:, 'Season':].drop(labels="CategoryPotential", axis=1)

            X_v2 = X.copy()
            X_v2.drop(["CR", "TopSKUBrand", "BrandChange"], axis=1, inplace=True)

            # 构建整体模型
            model = fit_model(X_v2, Y)
            print(r2_score(Y, model.predict(X_v2)))
            print(model.summary())
            with open(cate_saving_path + '_2_2整体回归模型解读.csv', 'w', encoding='utf-8') as f:
                f.write(model.summary().as_csv())

            # 输出整体模型共线性情况, vif>10时，共线性问题严重，Y的估计值可用，整体模型的回归系数和显著度待斟酌
            vif = get_vif(X)
            vif.to_csv(cate_saving_path + '_2_1共线性检验vif.csv', encoding='GBK')

            # 构建各维度得分计算模型，（若后续需要关系分析，建议从这些模型来，暂时排除共线性问题）
            user_model = fit_model(X[['User', 'UserRate', 'User_UserRate']], Y)
            #             print(user_model.params)
            X_user = X[['User', 'UserRate', 'User_UserRate']]
            with open(cate_saving_path + '_5_1整体user_model回归模型解读.csv', 'w', encoding='utf-8') as f:
                f.write(user_model.summary().as_csv())
            vif = get_vif(X_user)
            vif.to_csv(cate_saving_path + '_5_1user_model共线性检验vif.csv', encoding='GBK')

            traffic_model = fit_model(X[['Traffic', 'TrafficRate', 'Traffic_TrafficRate']], Y)
            #             print(traffic_model.params)
            X_traffic = X[['Traffic', 'TrafficRate', 'Traffic_TrafficRate']]
            with open(cate_saving_path + '_5_2整体traffic_model回归模型解读.csv', 'w', encoding='utf-8') as f:
                f.write(traffic_model.summary().as_csv())
            vif = get_vif(X_traffic)
            vif.to_csv(cate_saving_path + '_5_2traffic_model共线性检验vif.csv', encoding='GBK')

            sale_model = fit_model(X[['Sale', 'SaleRate', 'Sale_SaleRate']], Y)
            #             print(sale_model.params)
            X_sale = X[['Sale', 'SaleRate', 'Sale_SaleRate']]
            with open(cate_saving_path + '_5_3整体sale_model回归模型解读.csv', 'w', encoding='utf-8') as f:
                f.write(sale_model.summary().as_csv())
            vif = get_vif(X_sale)
            vif.to_csv(cate_saving_path + '_5_3sale_model共线性检验vif.csv', encoding='GBK')

            compete_model = fit_model(X[['CR', 'CRRate', 'TopSKUBrand', 'BrandChange', 'CR_SaleRate', 'CR_CRRate',
                                         'Sale_TopSKUBrand', 'Sale_BrandChange']], Y)
            #             print(compete_model.params)
            X_compete = X[['CR', 'CRRate', 'TopSKUBrand', 'BrandChange', 'CR_SaleRate', 'CR_CRRate',
                           'Sale_TopSKUBrand', 'Sale_BrandChange']]
            with open(cate_saving_path + '_5_4整体compete_model回归模型解读.csv', 'w', encoding='utf-8') as f:
                f.write(compete_model.summary().as_csv())
            vif = get_vif(X_compete)
            vif.to_csv(cate_saving_path + '_5_4compete_model共线性检验vif.csv', encoding='GBK')

            # 竞争力模型v2
            compete_model_v2 = fit_model(
                X[['CR_2', 'CRRate', 'TopSKUBrand_2', 'BrandChange_2', 'CR_SaleRate', 'CR_CRRate',
                   'Sale_TopSKUBrand', 'Sale_BrandChange']], Y)
            #             print(compete_model_v2.params)
            X_compete_v2 = X[['CR_2', 'CRRate', 'TopSKUBrand_2', 'BrandChange_2', 'CR_SaleRate', 'CR_CRRate',
                              'Sale_TopSKUBrand', 'Sale_BrandChange']]
            with open(cate_saving_path + '_5_5_v2_compete_model回归模型解读.csv', 'w', encoding='utf-8') as f:
                f.write(compete_model_v2.summary().as_csv())
            vif = get_vif(X_compete_v2)
            vif.to_csv(cate_saving_path + '_5_5compete_model_v2共线性检验vif.csv', encoding='GBK')

            # 改变index便于进行月份统计
            cate_df_train.reset_index(inplace=True)

            # 分月度统计R方并存储
            dt_list = set(cate_df_train["dt"])
            for dt in dt_list:
                # X_dt = cate_df_train[cate_df_train['dt'] == dt].loc[:, 'Season':].drop(labels='CategoryPotential',
                #                                                                        axis=1)
                X_dt = cate_df_train[cate_df_train['dt'] == dt].loc[:, 'Season':].drop(labels=["CR", "TopSKUBrand", "BrandChange",'CategoryPotential'],
                                                                                       axis=1)
                # X_dt.drop(["CR", "TopSKUBrand", "BrandChange"], axis=1, inplace=True)
                Y_dt = cate_df_train[cate_df_train['dt'] == dt]['CategoryPotential']
                with open(saving_path + r'/0_0模型解释度_R方.csv', "a", newline='', encoding='GBK') as f:
                    writer = csv.writer(f)
                    writer.writerow((cate, dt, r2_score(Y_dt, model.predict(X_dt))))

            with open(saving_path + r'/0_0模型解释度_R方.csv', "a", newline='', encoding='GBK') as f:
                writer = csv.writer(f)
                writer.writerow((cate, 'all', r2_score(Y, model.predict(X_v2))))

            # 因子整体得分模型
            with open(saving_path + r'/0_0模型解释度_R方.csv', "a", newline='', encoding='GBK') as f:
                writer = csv.writer(f)
                writer.writerow((cate, 'user', r2_score(Y, user_model.predict(X_user))))

            with open(saving_path + r'/0_0模型解释度_R方.csv', "a", newline='', encoding='GBK') as f:
                writer = csv.writer(f)
                writer.writerow((cate, 'traffic', r2_score(Y, traffic_model.predict(X_traffic))))

            with open(saving_path + r'/0_0模型解释度_R方.csv', "a", newline='', encoding='GBK') as f:
                writer = csv.writer(f)
                writer.writerow((cate, 'sale', r2_score(Y, sale_model.predict(X_sale))))

            with open(saving_path + r'/0_0模型解释度_R方.csv', "a", newline='', encoding='GBK') as f:
                writer = csv.writer(f)
                writer.writerow((cate, 'compete', r2_score(Y, compete_model.predict(X_compete))))
                writer.writerow((cate, 'compete_v2', r2_score(Y, compete_model_v2.predict(X_compete_v2))))
                # writer.writerow((cate, 'compete_change', r2_score(Y, compete_change.predict(X_change))))
                # writer.writerow((cate, 'compete_cover', r2_score(Y, compete_cover.predict(X_cover))))

            # 根据最新月份数据进行得分分析
            cate_df_final.reset_index(inplace=True)
            cate_df_new = cate_df_final[cate_df_final['dt'] == '2020-10'].copy()
            cate_df_new.drop_duplicates()
            X_new = cate_df_new.loc[:, 'Season':].drop(labels=["CR", "TopSKUBrand", "BrandChange",'CategoryPotential'], axis=1)
            # X_new = cate_df_new.loc[:, 'Season':].drop(labels='CategoryPotential', axis=1)
            # X_new.drop(["CR", "TopSKUBrand", "BrandChange"], axis=1, inplace=True)
            cate_df_new['e_CategoryPotential'] = model.predict(X_new).tolist()
            list_en = ['Season', 'BrandChange', 'CRRate', 'TopSKUBrand', 'CR', 'SaleRate',
                       'Sale', 'TrafficRate', 'Traffic', 'UserRate', 'User', 'User_UserRate',
                       'Traffic_TrafficRate', 'Sale_SaleRate', 'CR_SaleRate', 'CR_CRRate',
                       'Sale_TopSKUBrand', 'Sale_BrandChange', 'CR_2', "TopSKUBrand_2", 'BrandChange_2']
            cate_df_new = cate_df_new.drop_duplicates(subset=list_en, keep="first")  # 与顺序无关 按照值进行索引

            # month_1 = sorted(cate_df_final["dt"].unique())[-1:][0]
            # month_2 = sorted(cate_df_final["dt"].unique())[-2:][0]
            # month_3 = sorted(cate_df_final["dt"].unique())[-3:][0]
            #
            # ch = ["用户占比得分", "用户同比得分", "流量占比得分", "流量同比得分", "销售占比得分", "销售同比得分", "集中度占比得分", "集中度同比得分", "topsku店铺数得分",
            #       "店铺变动得分"]
            # en = ["User", "UserRate", "Traffic", "TrafficRate", "Sale", "SaleRate", "CR", "CRRate", "TopSKUBrand",
            #       "BrandChange"]
            # dict_latent = dict(zip(en, ch))
            # scores_vice = scores.copy()
            # scores_vice.reset_index(inplace=True)
            # latent_scores = scores_vice[(scores_vice['dt'] == month_1) | (scores_vice['dt'] == month_2) | (
            #         scores_vice['dt'] == month_3)].loc[:, :"User"]
            # latent_scores.drop("Season", axis=1, inplace=True)
            # latent_scores.rename(columns=dict_latent, inplace=True)
            # cate = cate.replace(r"/", "_")  # 家庭清洁/纸品  家庭清洁_纸品
            # latent_scores.to_csv(saving_data_test + r'/' + str(cate) + r"_latent_test_data.csv", index=False,
            #                      header=True)
            #
            # cate_df_new_3 = cate_df_final[
            #     (cate_df_final['dt'] == month_1) | (cate_df_final['dt'] == month_2) | (cate_df_final['dt'] == month_3)]
            # X_new_3 = cate_df_new_3.loc[:, 'Season':].drop(labels='CategoryPotential', axis=1)
            # X_index_df = pd.concat([cate_df_new_3.loc[:, ["dt", "item_last_cate_cd"]], X_new_3], axis=1)
            # X_index_df.drop_duplicates(keep="first").to_csv(saving_data_test + r'/' + str(cate) + r"_test_data.csv",
            #                                                 index=False, header=True)
            #
            # user_test_data = X_index_df.loc[:, ["dt", "item_last_cate_cd", 'User', 'UserRate', 'User_UserRate']]
            # user_test_data.drop_duplicates(keep="first").to_csv(
            #     saving_data_test + r'/' + str(cate) + r"_user_test_data.csv", index=False, header=True)
            #
            # sale_test_data = X_index_df.loc[:, ["dt", "item_last_cate_cd", 'Sale', 'SaleRate', 'Sale_SaleRate']]
            # sale_test_data.drop_duplicates(keep="first").to_csv(
            #     saving_data_test + r'/' + str(cate) + r"_sale_test_data.csv", index=False, header=True)
            #
            # traffic_test_data = X_index_df.loc[:,
            #                     ["dt", "item_last_cate_cd", 'Traffic', 'TrafficRate', 'Traffic_TrafficRate']]
            # traffic_test_data.drop_duplicates(keep="first").to_csv(
            #     saving_data_test + r'/' + str(cate) + r"_traffic_test_data.csv", index=False, header=True)
            #
            # compete_test_data = X_index_df.loc[:,
            #                     ["dt", "item_last_cate_cd", 'CR', 'CRRate', 'TopSKUBrand', 'BrandChange', 'CR_SaleRate',
            #                      'CR_CRRate', 'Sale_TopSKUBrand', 'Sale_BrandChange']]
            # compete_test_data.drop_duplicates(keep="first").to_csv(
            #     saving_data_test + r'/' + str(cate) + r"_compete_test_data.csv", index=False, header=True)

            # 计算得分
            cate_df_new['UserScore'] = user_model.predict(cate_df_new[['User', 'UserRate', 'User_UserRate']]).tolist()
            cate_df_new['TrafficScore'] = traffic_model.predict(
                cate_df_new[['Traffic', 'TrafficRate', 'Traffic_TrafficRate']]).tolist()
            cate_df_new['SaleScore'] = sale_model.predict(cate_df_new[['Sale', 'SaleRate', 'Sale_SaleRate']]).tolist()
            cate_df_new['CompeteScore'] = compete_model_v2.predict(
                cate_df_new[['CR_2', 'CRRate', 'TopSKUBrand_2', 'BrandChange_2',
                             'CR_SaleRate', 'CR_CRRate', 'Sale_TopSKUBrand',
                             'Sale_BrandChange']]).tolist()
            # 整理最终结果并输出
            sub_col = ['item_last_cate_cd',
                       'item_last_cate_name',
                       'dt',
                       'e_CategoryPotential',
                       '用户综合得分',
                       '流量综合得分',
                       '销售综合得分',
                       '竞争力综合得分',
                       '用户占比得分',
                       '用户同比得分',
                       '流量占比得分',
                       '流量同比得分',
                       '销售占比得分',
                       '销售同比得分',
                       '集中度占比得分',
                       '集中度同比得分',
                       'topsku店铺数得分',
                       '店铺变动得分']
            result_df = cate_df_new.loc[:, ['item_first_cate_cd', 'item_first_cate_name', 'item_second_cate_cd', \
                                            'item_second_cate_name', 'item_third_cate_cd', 'item_third_cate_name', \
                                            'item_fourth_cate_cd', 'item_fourth_cate_name', 'item_last_cate_cd', \
                                            'item_last_cate_name', 'dt', \
                                            'e_CategoryPotential', \
                                            'UserScore', 'TrafficScore', 'SaleScore', 'CompeteScore', \
                                            'User', 'UserRate', 'Traffic', 'TrafficRate', 'Sale', 'SaleRate', \
                                            'CR', 'CRRate', 'TopSKUBrand', 'BrandChange']]

            result_df.drop_duplicates(subset=sub_col, keep="first", inplace=True)

            result_df.set_index(
                ['item_first_cate_cd', 'item_first_cate_name', 'item_second_cate_cd', \
                 'item_second_cate_name', 'item_third_cate_cd', 'item_third_cate_name', \
                 'item_fourth_cate_cd', 'item_fourth_cate_name', 'item_last_cate_cd', \
                 'item_last_cate_name', 'dt'], inplace=True)

            # 开始归一化 不希望看到0 希望得到非零值
            result_df = (result_df - result_df.min()) * 100 / (result_df.max() - result_df.min())
            result_df_mean = result_df.mean()
            result_df = result_df + result_df_mean
            result_df_max = result_df.max()
            result_df = result_df * 100 / result_df_max
            dict_chinese = dict(zip(origin, chinese))
            result_df.rename(columns=dict_chinese, inplace=True)

            result_df.sort_values(by='e_CategoryPotential', ascending=False, inplace=True)

            # 结果输出
            result_df.to_csv(cate_saving_path + '_3_1最终输出结果.csv', encoding='GBK')
            print("类目 {} 建模成功".format(cate))
        else:

            continue
    #         except Exception as e:
    #             cates.append(cate)
    print("类目 {} 不满足建模条件,总计 {} 个".format(cates_features, len(cates_features)))

# Y_Y_4 = cate_df_final_save.iloc[:,[-19,-18,-5,-1]].dropna(axis=0, how="any")
# Y_Y_4.reset_index(inplace=True)
# Y_Y1 = cate_df_final_save.iloc[:,[-5,-1]].dropna(axis=0, how="any")
# Y_Y1.reset_index(inplace=True)
# mm = Y_Y1[(Y_Y1["dt"]=="2019-09")][["CategoryPotential", "CategoryPotential_score"]]
# mm.dropna(axis=0, how="any")
# mm_1 = mm.apply(lambda x: ((x - min(x))*100/(max(x)-min(x))))
# # plt.plot(mm_1["CategoryPotential"].values, "-",label="CategoryPotential")
# plt.plot(mm_1["CategoryPotential"].values, "-",label="CategoryPotential")
# plt.plot(-1*mm_1["CategoryPotential_score"].values+100, "--",label="CategoryPotential_score")
# plt.legend(loc="best")
# plt.title(cate+"CategoryPotential & CategoryPotential_score")
# plt.ylabel("values")
#
# # all_data_vali = cate_df_final_save.iloc[:,[-19,-18,-5,-1]].dropna(axis=0, how="any")
# cate_df_final_save.loc[("2020-03",21327)][['growth_pct', 'adjust_rate']]
# cate_df_final_save.loc[("2020-03",21327)][["CategoryPotential","CategoryPotential_score"]]
# "CategoryPotential","CategoryPotential_score"]
# Y_Y_4[Y_Y_4["CategoryPotential"]==-0.1797]

# Y_Y_4 = cate_df_final_save.iloc[:,[-19,-18,-5,-1]].dropna(axis=0, how="any")
# Y_Y_4.reset_index(inplace=True)
# Y_Y1 = cate_df_final_save.iloc[:,[-5,-1]].dropna(axis=0, how="any")
# Y_Y1.reset_index(inplace=True)
# mm = Y_Y1[(Y_Y1["dt"]=="2020-05")][["CategoryPotential", "CategoryPotential_score"]]
# mm.dropna(axis=0, how="any")
# mm_1 = mm.apply(lambda x: ((x - min(x))*100/(max(x)-min(x))))
# # plt.plot(mm_1["CategoryPotential"].values, "-",label="CategoryPotential")
# plt.plot(mm_1["CategoryPotential"].values, "-",label="CategoryPotential")
# plt.plot(mm_1["CategoryPotential_score"].values, "--",label="CategoryPotential_score")
# plt.legend(loc="best")
# plt.title(cate+"CategoryPotential & CategoryPotential_score")
# plt.ylabel("values")