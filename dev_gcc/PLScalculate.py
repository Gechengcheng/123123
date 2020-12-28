# coding=utf-8
# version: python3.6
# 作者：Yin Pan
# 创建时间：2020/09/17 11:00
# 潜力挖掘项目中，利用清洗好的底层数据，构建项目模型，尝试使用功能包PLSPM
# 目前无法进行moderation统计


import pandas as pd
import plspm.config as c
from sklearn.linear_model import LinearRegression
from plspm.scale import Scale
from plspm.mode import Mode
import csv
from plspm.scheme import Scheme
from plspm.plspm import Plspm


def score_path_matrix():  # 二次潜变量 得分
    structure = c.Structure()
    structure.add_path(["User", "UserRate", "Traffic", "TrafficRate", "Sale", "SaleRate", "CR", "TopSKUBrand", \
                        "CRRate", "BrandChange", "Season"], ["CategoryPotential"])
    return structure.path()


def score_add_lv(config):  # modeler数据 构造一次潜变量
    config.add_lv("User", Mode.A, c.MV("order_user"), c.MV("search_user"))
    config.add_lv("UserRate", Mode.A, c.MV("order_user_rate"), c.MV("search_user_rate"))
    config.add_lv("Traffic", Mode.A, c.MV("uv"), c.MV("pv"), c.MV("search"))
    config.add_lv("TrafficRate", Mode.A, c.MV("uv_rate"), c.MV("pv_rate"), c.MV("search_rate"))
    config.add_lv("Sale", Mode.A, c.MV("sale"), c.MV("gmv"))
    config.add_lv("SaleRate", Mode.A, c.MV("sale_rate"), c.MV("gmv_rate"))
    config.add_lv("CR", Mode.A, c.MV("cr3"), c.MV("cr5"), c.MV("cr8"))
    config.add_lv("TopSKUBrand", Mode.A, c.MV("top10brand"), c.MV("top20brand"), c.MV("top50brand"))
    config.add_lv("CRRate", Mode.A, c.MV("cr3_rate"), c.MV("cr5_rate"), c.MV("cr8_rate"))
    config.add_lv("BrandChange", Mode.A, c.MV("brand_change3"), c.MV("brand_change5"), c.MV("brand_change8"))
    config.add_lv("Season", Mode.B, c.MV("month", Scale.NOM))  # 数据集较小，时间序列方法不适用，目前仅观察一级品类是否有特殊月份起伏
    config.add_lv("CategoryPotential", Mode.B, c.MV("growth_pct"), c.MV("adjust_rate"))  # Mode.B 进行formative测量


if __name__ == "__main__":
    file_path = r'D:\gechengcheng3\Desktop\潜力类目挖掘\test_data.csv'
    saving_path = r'D:\gechengcheng3\Desktop\潜力类目挖掘\result'
    df = pd.read_csv(file_path, delimiter='\t', index_col=['dt', 'item_third_cate_cd'])
    with open(saving_path + r'\R_result.csv', "w", newline='', encoding='GBK') as f:
        writer = csv.writer(f)
        writer.writerow(('cate', 'dt', 'R_2'))  # 双指标？一行一行写入
    # 读取所有一级类目列表
    cate_list = set(df["item_first_cate_name"])
    # 按照一级类目进行循环
    for cate in cate_list:
        cate_saving_path = saving_path + '\\' + cate.replace('/', '')

        cate_df = df[df["item_first_cate_name"] == cate]
        cate_df_filled = cate_df.fillna(cate_df.mean())  # 利用均值填充缺失值(可能存在一定风险)进行潜变量计算

        # 定义路径和度量模型
        config = c.Config(score_path_matrix(), default_scale=Scale.NUM)
        score_add_lv(config)
        # 进行潜变量计算
        plspm_calc = Plspm(cate_df_filled, config, Scheme.PATH, iterations=1000, tolerance=1e-06)
        scores = plspm_calc.scores()
        # 潜变量得分 与 原始矩阵合并
        cate_df_final = pd.merge(cate_df, scores, how='left', left_index=True, right_index=True,
                                 suffixes=('', '_latent'))
        # 准备数据进行多元回归分析
        # 生成调节变量
        mod_list = [('User', 'UserRate'), ('Traffic', 'TrafficRate'), ('Sale', 'SaleRate'), ('CR', 'SaleRate'),
                    ('CR', 'CRRate'), ('Sale', 'TopSKUBrand'), ('Sale', 'BrandChange')]  # 少了两个？对比输出结果的Excel表
        for mod in mod_list:
            cate_df_final[str(mod[0] + '_' + mod[1])] = cate_df_final[mod[0]] * cate_df_final[mod[1]]
        cate_df_train = cate_df_final.dropna(axis=0, how='any')  # 仅利用完全非空行进行统计 即全部不为空的记录行 对应all

        # 构建训练数据
        Y = cate_df_train['CategoryPotential']
        X = cate_df_train.loc[:, 'Season':].drop(labels='CategoryPotential', axis=1)

        # 构建整体模型
        model = LinearRegression()
        model.fit(X, Y)

        # 计算用户指数模型
        user_model = LinearRegression()
        user_model.fit(X[['User', 'UserRate', 'User_UserRate']], Y)
        # 计算流量指数模型
        traffic_model = LinearRegression()
        traffic_model.fit(X[['Traffic', 'TrafficRate', 'Traffic_TrafficRate']], Y)
        # 计算销量指数模型
        sale_model = LinearRegression()
        sale_model.fit(X[['Sale', 'SaleRate', 'Sale_SaleRate']], Y)
        # 计算市场竞争指数模型
        compete_model = LinearRegression()
        compete_model.fit(X[['CR', 'CRRate', 'TopSKUBrand', 'BrandChange', 'CR_SaleRate', 'CR_CRRate',
                             'Sale_TopSKUBrand', 'Sale_BrandChange']], Y)

        cate_df_train.reset_index(inplace=True)
        # 分月度统计R方并存储
        dt_list = set(cate_df_train["dt"])
        for dt in dt_list:
            X_dt = cate_df_train[cate_df_train['dt'] == dt].loc[:, 'Season':].drop(labels='CategoryPotential', axis=1)
            Y_dt = cate_df_train[cate_df_train['dt'] == dt]['CategoryPotential']
            with open(saving_path + r'\R_result.csv', "a", newline='', encoding='GBK') as f:
                writer = csv.writer(f)
                writer.writerow((cate, dt, model.score(X_dt, Y_dt)))
        with open(saving_path + r'\R_result.csv', "a", newline='', encoding='GBK') as f:
            writer = csv.writer(f)
            writer.writerow((cate, 'all', model.score(X, Y)))

        # 根据最新月份数据进行得分分析
        cate_df_final.reset_index(inplace=True)
        cate_df_new = cate_df_final[cate_df_final['dt'] == '2020-07']
        X_new = cate_df_new.loc[:, 'Season':].drop(labels='CategoryPotential', axis=1)
        cate_df_new['e_CategoryPotential'] = model.predict(X_new).tolist()
        # print(cate_df_new.info())

        # 计算得分
        cate_df_new['UserScore'] = user_model.predict(cate_df_new[['User', 'UserRate', 'User_UserRate']]).tolist()
        cate_df_new['TrafficScore'] = traffic_model.predict(
            cate_df_new[['Traffic', 'TrafficRate', 'Traffic_TrafficRate']]).tolist()
        cate_df_new['SaleScore'] = sale_model.predict(cate_df_new[['Sale', 'SaleRate', 'Sale_SaleRate']]).tolist()
        cate_df_new['CompeteScore'] = compete_model.predict(cate_df_new[['CR', 'CRRate', 'TopSKUBrand', 'BrandChange',
                                                                         'CR_SaleRate', 'CR_CRRate', 'Sale_TopSKUBrand',
                                                                         'Sale_BrandChange']]).tolist()

        result_df = cate_df_new.loc[:, ['item_first_cate_cd', 'item_first_cate_name', 'item_second_cate_cd',
                                        'item_second_cate_name', 'item_third_cate_cd', 'item_third_cate_name', 'dt',
                                        'e_CategoryPotential',
                                        'UserScore', 'TrafficScore', 'SaleScore', 'CompeteScore',
                                        'User', 'UserRate', 'Traffic', 'TrafficRate', 'Sale', 'SaleRate',
                                        'CR', 'CRRate', 'TopSKUBrand', 'BrandChange', 'Season']]
        result_df.set_index(
            ['item_first_cate_cd', 'item_first_cate_name', 'item_second_cate_cd', 'item_second_cate_name',
             'item_third_cate_cd', 'item_third_cate_name', 'dt'], inplace=True)

        # 开始归一化
        result_df = (result_df - result_df.min()) * 100 / (result_df.max() - result_df.min())
        result_df.sort_values(by='e_CategoryPotential', ascending=False, inplace=True)

        # 结果输出
        result_df.to_csv(cate_saving_path + '_final_result.csv', encoding='GBK')
        print(cate, cate_df_final.shape[0], cate_df.shape[0])



