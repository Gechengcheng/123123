c2m的文本数据提取和清洗的方式方法

文本数据特征提取

文本数据清洗

11-05
堡垒机中直接执行：
hive -f biaoming,sql

或者 将"hive -f biaoming.sql"写入 star.sh
执行bash star.sh 即可

hive中对双引号的限制和Python中对双引号的限制不同
hive中限制的不严格但是在Python中限制的很严格
hive中的日期可以使用双引号或者单引号执行 但是Python中的sql 只能使用单引号去引用日期


一级二级三级 spu  sku 自营 sku spu 一对一
一级二级三级 item_id sku POP

情感分析模型梳理：




无需对显变量进行标准化  计算出潜变量之后 自动标准化处理 均值为0 和标注差为1

模型预测保存的结果表是带有一二三级类目和dt的结果表

无论是sm和statsModels都是用的潜变量都是归一化之后的数据

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
os.chdir(r'D:\gechengcheng3\Desktop\潜力类目挖掘')


def score_path_matrix():
    structure = c.Structure()
    structure.add_path(["User", "UserRate", "Traffic", "TrafficRate", "Sale", "SaleRate", "CR", "TopSKUBrand", \
                        "CRRate", "BrandChange", "Season"], ["CategoryPotential"])
    return structure.path()


def score_add_lv(config):
    config.add_lv("User", Mode.A, c.MV("order_user"), c.MV("search_user"))
    config.add_lv("UserRate", Mode.A, c.MV("order_user_rate"), c.MV("search_user_rate"))
    config.add_lv("Traffic", Mode.A, c.MV("pv"), c.MV("search"))
    config.add_lv("TrafficRate", Mode.A, c.MV("pv_rate"), c.MV("search_rate"))
    config.add_lv("Sale", Mode.A, c.MV("sale"), c.MV("gmv"))
    config.add_lv("SaleRate", Mode.A, c.MV("sale_rate"), c.MV("gmv_rate"))
    config.add_lv("CR", Mode.A, c.MV("cr3"), c.MV("cr5"), c.MV("cr8"))
    config.add_lv("TopSKUBrand", Mode.A, c.MV("top10brand"), c.MV("top20brand"), c.MV("top50brand"))
    config.add_lv("CRRate", Mode.A, c.MV("cr3_rate"), c.MV("cr5_rate"), c.MV("cr8_rate"))
    config.add_lv("BrandChange", Mode.A, c.MV("brand_change3"), c.MV("brand_change5"), c.MV("brand_change8"))
    config.add_lv("Season", Mode.B, c.MV("month", Scale.NOM))  # 数据集较小，时间序列方法不适用，目前仅观察一级品类是否有特殊月份起伏
    config.add_lv("CategoryPotential", Mode.B, c.MV("growth_pct"), c.MV("adjust_rate"))  # Mode.B 进行formative测量


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

def fit_model(X_df, Y_df):
    this_model = sm.OLS(Y_df, X_df).fit()
    return this_model


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_dir', type=str, default='/media/cfs/gechengcheng3/test_poten/poten/',
    #                     help='set model saving path')
    #
    # parser.add_argument('--input_data', type=str,
    #                     default='/media/cfs/gechengcheng3/test_poten/poten/data_test_5_first_cate.txt',
    #                     help='set input data path')

    # args = parser.parse_args()

    model_dir = r'model_save'
    file_path = r'data_test_5_first_cate.txt'

    df_cate = pd.read_csv(file_path, delimiter='\t', index_col=['dt', 'item_third_cate_cd'], encoding="gbk")

    # 预处理step1检测并去除重复行
    for i in np.array(df.columns[5:]):
        drop_duplicate(df, i)
    print('数据集是否存在重复观测: ', any(df.duplicated()))
    print('\n重复值处理完毕\n')

    cate_list = set(df["item_first_cate_cd"])
    cates = []
    for cate in cate_list:
        try:

            cate_df = df[df["item_first_cate_cd"] == cate]

            data_final_float_type = list(cate_df.dtypes.loc[cate_df.dtypes.values == 'float64'].index)
            cate_df = outer_deal(cate_df, data_final_float_type)

            print(cate_df.shape)
            print(cate)
            cate_df_filled = cate_df.fillna(cate_df.mean())

            # 定义路径和度量模型
            config = c.Config(score_path_matrix(), default_scale=Scale.NUM)
            score_add_lv(config)
            # 进行潜变量计算
            plspm_calc = Plspm(cate_df_filled, config, Scheme.PATH, iterations=1000, tolerance=1e-06, bootstrap=False)
            scores = plspm_calc.scores()

            # 潜变量得分 与 原始矩阵合并
            cate_df_final = pd.merge(cate_df, scores, how='left', left_index=True, right_index=True,
                                     suffixes=('', '_latent'))
            # 生成调节变量
            mod_list = [('User', 'UserRate'), ('Traffic', 'TrafficRate'), ('Sale', 'SaleRate'), ('CR', 'SaleRate'),
                        ('CR', 'CRRate'), ('Sale', 'TopSKUBrand'), ('Sale', 'BrandChange')]
            for mod in mod_list:
                cate_df_final[str(mod[0] + '_' + mod[1])] = cate_df_final[mod[0]] * cate_df_final[mod[1]]

            cate_df_train = cate_df_final.dropna(axis=0, how='any')  # 仅利用非空行进行统计
            print(cate_df_train.shape[0], cate_df_final.shape[0])

            # 构建训练数据
            Y = cate_df_train['CategoryPotential']
            X = cate_df_train.loc[:, 'Season':].drop(labels='CategoryPotential', axis=1)

            # 建立模型
            model = linear_model.LinearRegression(fit_intercept=False)
            model.fit(X, Y)
            print("总模型系数:", model.coef_, '\n', model.intercept_)
            print('finish training')
            print("sklearn预测结果", model.predict(X[:5]))

            model_sm = fit_model(X, Y)
            print(model_sm.params)
            print(model_sm.predict(X[:5]))

            # 建立综合得分总模型
            model_user = linear_model.LinearRegression(fit_intercept=False)
            model_user.fit(X[['User', 'UserRate', 'User_UserRate']], Y)

            model_traffic = linear_model.LinearRegression(fit_intercept=False)
            model_traffic.fit(X[['Traffic', 'TrafficRate', 'Traffic_TrafficRate']], Y)

            model_sale = linear_model.LinearRegression(fit_intercept=False)
            model_sale.fit(X[['Sale', 'SaleRate', 'Sale_SaleRate']], Y)

            model_compete = linear_model.LinearRegression(fit_intercept=False)
            model_compete.fit(X[['CR', 'CRRate', 'TopSKUBrand', 'BrandChange', 'CR_SaleRate', 'CR_CRRate',
                                 'Sale_TopSKUBrand', 'Sale_BrandChange']], Y)

            # save the model
            # joblib.dump(model, model_dir + str(cate) + 'REGRESSOR.joblib')
            # joblib.dump(model_user, model_dir + str(cate) + 'user.joblib')
            # joblib.dump(model_traffic, model_dir + str(cate) + 'traffic.joblib')
            # joblib.dump(model_sale, model_dir + str(cate) + 'sale.joblib')
            # joblib.dump(model_compete, model_dir + str(cate) + 'compete.joblib')

            # print('model saved')

        except Exception as e:
            cates.append(cate)
    print(cates)

# 以上代码验证 在数据标准化之后 sklearn和sm得到的结果是一样的  无论是预测结果还是系数都是完全一致的













