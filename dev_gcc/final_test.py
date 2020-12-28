# coding=utf-8
# version: python3.6
# 作者：Yin Pan
# 创建时间：2020/09/17 11:00
# 潜力挖掘项目中，利用清洗好的底层数据，构建项目模型，并且输出最终结果
# 结构方程包参考资料：https://github.com/GoogleCloudPlatform/plspm-python；https://plspm.readthedocs.io/en/latest/

import numpy as np
import joblib
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
import os
os.chdir(r"D:\gechengcheng3\Desktop\潜力类目挖掘")
file_path = r'data_test_5_first_cate.txt'
saving_path = r'rest_1026_2_3'
df = pd.read_csv(file_path, delimiter='\t', index_col=['dt', 'item_third_cate_cd'], encoding="gbk")
def score_path_matrix():
    structure = c.Structure()
    structure.add_path(["User", "UserRate", "Traffic", "TrafficRate", "Sale", "SaleRate", "CR", "TopSKUBrand", \
                        "CRRate", "BrandChange", "Season"], ["CategoryPotential"])
    return structure.path()


def score_add_lv(config):
    config.add_lv("User", Mode.A, c.MV("order_user"), c.MV("search_user"))
    config.add_lv("UserRate", Mode.A, c.MV("order_user_rate"), c.MV("search_user_rate"))
    config.add_lv("Traffic", Mode.A, c.MV("pv"),  c.MV("uv"), c.MV("search"))
    config.add_lv("TrafficRate", Mode.A, c.MV("pv_rate"),  c.MV("uv_rate"), c.MV("search_rate"))
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
           'item_third_cate_cd', 'item_third_cate_name', 'dt', \
           "e_CategoryPotential", "用户综合得分", "流量综合得分", "销售综合得分", "竞争力综合得分", "用户占比得分", "用户同比得分", "流量占比得分", "流量同比得分", \
           "销售占比得分", "销售同比得分", "集中度占比得分", "集中度同比得分", "topsku店铺数得分", "店铺变动得分"]

origin = ['item_first_cate_cd', 'item_first_cate_name', 'item_second_cate_cd', \
          'item_second_cate_name', 'item_third_cate_cd', 'item_third_cate_name', 'dt', \
          'e_CategoryPotential', \
          'UserScore', 'TrafficScore', 'SaleScore', 'CompeteScore', \
          'User', 'UserRate', 'Traffic', 'TrafficRate', 'Sale', 'SaleRate', \
          'CR', 'CRRate', 'TopSKUBrand', 'BrandChange']


def drop_duplicate(df_cate, x):
    """
    重复值处理，去除每一个特征中重复值高于百分之95的特征
    """
    if ((df_cate[x].value_counts().sort_values(ascending=False).max() / df_cate[x].value_counts().sum()) > 0.95):
        print('去除的特征为{0}，该特征内重复值占比为{1:.4f}\n'.format(x, (
                df_cate[x].value_counts().sort_values(ascending=False).max() / df_cate[x].value_counts().sum())))
        del df_cate[x]


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
            # print(Q3 + 1.5 * IQR)
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



if __name__ == "__main__":

    file_path = r'data_7_final_index_all.txt'
    saving_path = r'2339'
    df = pd.read_csv(file_path, delimiter='\t', index_col=['dt', 'item_third_cate_cd'], encoding="utf8")

    # 预处理step1检测并去除重复行
    for i in np.array(df.columns[5:]):
        drop_duplicate(df, i)
    print('数据集是否存在重复观测: ', any(df.duplicated()))
    print('\n重复值处理完毕\n')

    # # 预处理step2异常值处理
    # data_final_float_type = list(df.dtypes.loc[df.dtypes.values == 'float64'].index)
    # df = outer_deal(df, data_final_float_type)

    # 标准化数据参与潜变量计算和最终的模型拟合 来自业务输入
    with open(saving_path + r'\0_0模型解释度_R方.csv', "w", newline='', encoding='GBK') as f:
        writer = csv.writer(f)
        writer.writerow(('cate', 'dt', 'R_2'))
    # 读取所有一级类目列表 每一个一级类目拟合一个模型
    cate_list = set(df["item_first_cate_name"])
    # 按照一级类目进行循环

    cates = []  # 报错类目回溯矩阵记录
    for cate in cate_list:  # 每一个一级类目拟合一个模型
        # try:
            cate_saving_path = saving_path + '\\' + cate.replace('/', '')
            cate_df = df[df["item_first_cate_name"] == cate]

            # 预处理step2异常值处理
            data_final_float_type = list(cate_df.dtypes.loc[cate_df.dtypes.values == 'float64'].index)
            cate_df = outer_deal(cate_df, data_final_float_type)
            print(cate)

            cate_df_filled = cate_df.fillna(cate_df.mean())  # 利用均值填充缺失值(可能存在一定风险)进行潜变量计算,数据的时间跨度越宽风险越小 否则风险越大

            # 定义路径和度量模型
            config = c.Config(score_path_matrix(), default_scale=Scale.NUM)
            score_add_lv(config)
            # 进行潜变量计算
            plspm_calc = Plspm(cate_df_filled, config, Scheme.PATH, iterations=1000, tolerance=1e-06, bootstrap=False)
            scores = plspm_calc.scores()
            # scores.corr().iloc[-1:].to_csv(cate_saving_path + '_1_7潜力变量_其他潜变量corr.csv', encoding='GBK')
            # weights = plspm_calc.bootstrap().weights()
            # weights.to_csv(cate_saving_path + '_1_6weights结果.csv', encoding='GBK')

            # 输出潜变量指标
            # plspm_calc.inner_summary().to_csv(cate_saving_path + '_1_1潜变量内模型指标.csv', encoding='GBK')
            # plspm_calc.unidimensionality().to_csv(cate_saving_path + '_1_2潜变量外模型指标.csv', encoding='GBK') # only meaningful for reflective / mode A blocks

            # 潜变量直接关系系数 潜变量和显变量关系系数
            plspm_calc.inner_model().to_csv(cate_saving_path + '_1_3潜变量内模型.csv', encoding='GBK')
            plspm_calc.outer_model().to_csv(cate_saving_path + '_1_4潜变量外模型.csv', encoding='GBK')
            # scores.corr().to_csv(cate_saving_path + '_1_5潜变量相关系数.csv', encoding='GBK')  # 指标区分性
            # 潜变量得分 与 原始矩阵合并
            cate_df_final = pd.merge(cate_df, scores, how='left', left_index=True, right_index=True,
                                     suffixes=('', '_latent'))

            # cate_df_final.to_csv(cate_saving_path + "hs.csv", encoding='GBK')
            # 准备数据进行多元回归分析
            # 生成调节变量
            mod_list = [('User', 'UserRate'), ('Traffic', 'TrafficRate'), ('Sale', 'SaleRate'), ('CR', 'SaleRate'),
                        ('CR', 'CRRate'), ('Sale', 'TopSKUBrand'), ('Sale', 'BrandChange')]
            for mod in mod_list:
                cate_df_final[str(mod[0] + '_' + mod[1])] = cate_df_final[mod[0]] * cate_df_final[mod[1]]

            cate_df_train = cate_df_final.dropna(axis=0, how='any')  # 仅利用非空行进行统计,测试集合不参与训练
            print(cate_df_train.shape[0], cate_df_final.shape[0])
            # 构建训练数据
            Y = cate_df_train['CategoryPotential']
            X = cate_df_train.loc[:, 'Season':].drop(labels='CategoryPotential', axis=1)

            # 构建整体模型
            model = fit_model(X, Y)
            print(r2_score(Y, model.predict(X)))
            print(model.summary())
            with open(cate_saving_path + '_2_2整体回归模型解读.csv', 'w', encoding='utf-8') as f:
                f.write(model.summary().as_csv())

            # 输出整体模型共线性情况, vif>10时，共线性问题严重，Y的估计值可用，整体模型的回归系数和显著度待斟酌
            vif = get_vif(X)
            vif.to_csv(cate_saving_path + '_2_1共线性检验vif.csv', encoding='GBK')

            # 构建各维度得分计算模型，（若后续需要关系分析，建议从这些模型来，暂时排除共线性问题）
            user_model = fit_model(X[['User', 'UserRate', 'User_UserRate']], Y)
            # print(user_model.params)
            X_user = X[['User', 'UserRate', 'User_UserRate']]
            with open(cate_saving_path + '_5_1整体user_model回归模型解读.csv', 'w', encoding='utf-8') as f:
                f.write(user_model.summary().as_csv())

            traffic_model = fit_model(X[['Traffic', 'TrafficRate', 'Traffic_TrafficRate']], Y)
            # print(traffic_model.params)
            X_traffic = X[['Traffic', 'TrafficRate', 'Traffic_TrafficRate']]
            with open(cate_saving_path + '_5_2整体traffic_model回归模型解读.csv', 'w', encoding='utf-8') as f:
                f.write(traffic_model.summary().as_csv())

            sale_model = fit_model(X[['Sale', 'SaleRate', 'Sale_SaleRate']], Y)
            # print(sale_model.params)
            X_sale = X[['Sale', 'SaleRate', 'Sale_SaleRate']]
            with open(cate_saving_path + '_5_3整体sale_model回归模型解读.csv', 'w', encoding='utf-8') as f:
                f.write(sale_model.summary().as_csv())

            compete_model = fit_model(X[['CR', 'CRRate', 'TopSKUBrand', 'BrandChange', 'CR_SaleRate', 'CR_CRRate',
                                         'Sale_TopSKUBrand', 'Sale_BrandChange']], Y)
            # print(compete_model.params)
            X_compete = X[['CR', 'CRRate', 'TopSKUBrand', 'BrandChange', 'CR_SaleRate', 'CR_CRRate',
                                         'Sale_TopSKUBrand', 'Sale_BrandChange']]
            with open(cate_saving_path + '_5_4整体compete_model回归模型解读.csv', 'w', encoding='utf-8') as f:
                f.write(compete_model.summary().as_csv())

            # 改变index便于进行月份统计
            cate_df_train.reset_index(inplace=True)

            # 分月度统计R方并存储
            dt_list = set(cate_df_train["dt"])
            for dt in dt_list:
                X_dt = cate_df_train[cate_df_train['dt'] == dt].loc[:, 'Season':].drop(labels='CategoryPotential', axis=1)
                Y_dt = cate_df_train[cate_df_train['dt'] == dt]['CategoryPotential']
                with open(saving_path + r'\0_0模型解释度_R方.csv', "a", newline='', encoding='GBK') as f:
                    writer = csv.writer(f)
                    writer.writerow((cate, dt, r2_score(Y_dt, model.predict(X_dt))))
            with open(saving_path + r'\0_0模型解释度_R方.csv', "a", newline='', encoding='GBK') as f:
                writer = csv.writer(f)
                writer.writerow((cate, 'all', r2_score(Y, model.predict(X))))

            # 因子整体得分模型
            with open(saving_path + r'\0_0模型解释度_R方.csv', "a", newline='', encoding='GBK') as f:
                writer = csv.writer(f)
                writer.writerow((cate, 'user', r2_score(Y, user_model.predict(X_user))))

            with open(saving_path + r'\0_0模型解释度_R方.csv', "a", newline='', encoding='GBK') as f:
                writer = csv.writer(f)
                writer.writerow((cate, 'traffic', r2_score(Y, traffic_model.predict(X_traffic))))

            with open(saving_path + r'\0_0模型解释度_R方.csv', "a", newline='', encoding='GBK') as f:
                writer = csv.writer(f)
                writer.writerow((cate, 'sale', r2_score(Y, sale_model.predict(X_sale))))

            with open(saving_path + r'\0_0模型解释度_R方.csv', "a", newline='', encoding='GBK') as f:
                writer = csv.writer(f)
                writer.writerow((cate, 'compete', r2_score(Y, compete_model.predict(X_compete))))

            # 根据最新月份数据进行得分分析
            cate_df_final.reset_index(inplace=True)
            cate_df_new = cate_df_final[cate_df_final['dt'] == '2020-10']
            X_new = cate_df_new.loc[:, 'Season':].drop(labels='CategoryPotential', axis=1)
            cate_df_new['e_CategoryPotential'] = model.predict(X_new).tolist()  # 可以将list直接赋值给dataframe的某一列

            # 计算得分
            cate_df_new['UserScore'] = user_model.predict(cate_df_new[['User', 'UserRate', 'User_UserRate']]).tolist()
            cate_df_new['TrafficScore'] = traffic_model.predict(
                cate_df_new[['Traffic', 'TrafficRate', 'Traffic_TrafficRate']]).tolist()

            cate_df_new['SaleScore'] = sale_model.predict(cate_df_new[['Sale', 'SaleRate', 'Sale_SaleRate']]).tolist()
            cate_df_new['CompeteScore'] = compete_model.predict(cate_df_new[['CR', 'CRRate', 'TopSKUBrand', 'BrandChange',
                                                                             'CR_SaleRate', 'CR_CRRate', 'Sale_TopSKUBrand',
                                                                            'Sale_BrandChange']]).tolist()

            # 因子特征数据


            # 整理最终结果并输出
            result_df = cate_df_new.loc[:, ['item_first_cate_cd', 'item_first_cate_name', 'item_second_cate_cd',\
                                            'item_second_cate_name', 'item_third_cate_cd', 'item_third_cate_name', 'dt',\
                                            'e_CategoryPotential',\
                                             'UserScore', 'TrafficScore', 'SaleScore', 'CompeteScore',\
                                             'User', 'UserRate', 'Traffic', 'TrafficRate', 'Sale', 'SaleRate',\
                                             'CR', 'CRRate', 'TopSKUBrand', 'BrandChange']]

            result_df.set_index(
                ['item_first_cate_cd', 'item_first_cate_name', 'item_second_cate_cd', 'item_second_cate_name',
                 'item_third_cate_cd', 'item_third_cate_name', 'dt'], inplace=True)

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
            result_df.corr().head(1).to_csv(cate_saving_path + '_4_1相关系数.csv', encoding='GBK')
            print(cate, cate_df_final.shape[0], cate_df.shape[0])
            # if cate=="服饰内衣":
            #     break
    #
    #     except Exception as e:
    #         cates.append(cate)
    # print(cates)

#
# month_1 = cate_df_final["dt"].unique()[-1:][0]
# month_2 = cate_df_final["dt"].unique()[-2:][0]
# month_3 = cate_df_final["dt"].unique()[-3:][0]
# ch = ["用户占比得分", "用户同比得分", "流量占比得分", "流量同比得分", "销售占比得分", "销售同比得分", "集中度占比得分", "集中度同比得分", "topsku店铺数得分","店铺变动得分"]
# en = ["User", "UserRate", "Traffic", "TrafficRate", "Sale", "SaleRate", "CR", "CRRate", "TopSKUBrand", "BrandChange"]
# dict_latent = dict(zip(en, ch))
# scores_vice = scores.copy()
# scores_vice.reset_index(inplace=True)
# latent_scores = scores_vice[(cate_df_final['dt'] == month_1) | (cate_df_final['dt'] == month_2) | (cate_df_final['dt'] == month_3)].loc[:, :"User"]
# latent_scores.drop("Season", axis=1, inplace=True)
# latent_scores.rename(columns=dict_latent, inplace=True)
# # # latent_scores.to_csv( , index=False, header=True)
#
# cate_df_new_3 = cate_df_final[(cate_df_final['dt'] == month_1) | (cate_df_final['dt'] == month_2)|(cate_df_final['dt'] == month_3)]
# X_new_3 = cate_df_new_3.loc[:, 'Season':].drop(labels='CategoryPotential', axis=1)
# X_index_df = pd.concat([cate_df_new_3.loc[:, ["dt", "item_third_cate_cd"]], X_new_3], axis=1)
# index_df_pre = cate_df_new_3.loc[:, ["dt", "item_third_cate_cd"]]
# # X_new_3.to_csv(r"test_data.csv", index=False, header=False)
# X_index_df.to_csv(r"test_data_index.csv", index=False, header=True)
# model.predict(X_new_3).tolist()
#
# joblib.dump(model, "all_model.pkl")
# model_load = joblib.load("all_model.pkl")
# x_test = pd.read_csv(r"test_data_index.csv", header=0)
# index_df_pre["result"] = model_load.predict(x_test.iloc[:, 2:]).tolist()
# # model_load.predict(x_test).tolist()
#
# joblib.dump(user_model, "user_model.pkl")
from sklearn import linear_model
# user_load = joblib.load("1320REGRESSOR.joblib")
#
# user_test_data = X_index_df.loc[:, ["dt", "item_third_cate_cd", 'User', 'UserRate', 'User_UserRate']]
# user_test_data.to_csv(r"user_test_data.csv", index=False, header=True)
#
# # user_test_data_1 = pd.read_csv(r"user_test_data.csv", header=0) # header 务必设置为0 设置为1时 使用的是 自然数作为列索引
#
# sale_test_data = X_index_df.loc[:, ["dt", "item_third_cate_cd", 'Sale', 'SaleRate', 'Sale_SaleRate']]
# sale_test_data.to_csv(r"sale_test_data.csv", index=False, header=True)
#
# traffic_test_data = X_index_df.loc[:, ["dt", "item_third_cate_cd", 'Traffic', 'TrafficRate', 'Traffic_TrafficRate']]
# traffic_test_data.to_csv(r"traffic_test_data.csv", index=False, header=True)
#
# compete_test_data = X_index_df.loc[:, ["dt", "item_third_cate_cd", 'CR', 'CRRate', 'TopSKUBrand', 'BrandChange', 'CR_SaleRate', 'CR_CRRate', 'Sale_TopSKUBrand', 'Sale_BrandChange']]
# compete_test_data.to_csv(r"compete_test_data.csv", index=False, header=True)
#
# index_df_pre["sale_score"] = user_load.predict(user_test_data_1.iloc[:, 2:]).tolist()
#
#
# mm=zip([1,2,3],[4,5,6],[7,8,9])
# for x ,y,z in mm:
#     print(x,y,z)
