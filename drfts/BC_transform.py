# BC变换暂时不是第一优先级

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import os
os.chdir(r"D:\gechengcheng3\Desktop\潜力类目挖掘")

data_zz = pd.read_csv(r"data_zz.txt", sep="\t", encoding="utf8")
data_jx = pd.read_csv(r"data_jx.txt", sep="\t", encoding="utf8")



# 月份覆盖
month_cover = list(data_jx["dt"].unique())

index_col = ["dt", "item_first_cate_name", "item_last_cate_cd"]

compete = ["top10_brand_nums", "top20_brand_nums", "top50_brand_nums",\
        "top3_index", "top5_index", "top8_index",\
        "cr3_rate", "cr5_rate", "cr8_rate",\
        "cr3", "cr5", "cr8"]  # 12
user = ["user_pct", "order_pct", "order_rate", "user_rate"]  # 4
traffic = ["pv_pct", "pv_rate", "search_pct", "search_rate"]  # 4
sale = ["sale_cnt_pct", "sale_cnt_rate", "gmv_pct", "gmv_rate"]  # 4

# 获取数据
# user_model
jx_user = data_jx.loc[:, index_col+user].copy()
zz_user = data_zz.loc[:, index_col+user].copy()
# traffic_model
jx_traffic = data_jx.loc[:, index_col+traffic].copy()
zz_traffic = data_zz.loc[:, index_col+traffic].copy()
# sale_model
jx_sale = data_jx.loc[:, index_col+sale].copy()
zz_sale = data_zz.loc[:, index_col+sale].copy()
# compete_model
jx_compete = data_jx.loc[:, index_col+compete].copy()
zz_compete = data_zz.loc[:, index_col+compete].copy()

# 统计小于0的变量
(jx_user.iloc[:,3:]<0).astype(int).sum(axis=0),
(jx_sale.iloc[:,3:]<0).astype(int).sum(axis=0),  #1 少
(jx_traffic.iloc[:,3:]<0).astype(int).sum(axis=0),
(jx_compete.iloc[:,3:]<0).astype(int).sum(axis=0)  #1 少

(zz_user.iloc[:,3:]<0).astype(int).sum(axis=0),
(zz_sale.iloc[:,3:]<0).astype(int).sum(axis=0),  # 1
(zz_traffic.iloc[:,3:]<0).astype(int).sum(axis=0),
(zz_compete.iloc[:,3:]<0).astype(int).sum(axis=0)  # 1






