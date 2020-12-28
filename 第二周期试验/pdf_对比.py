import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import os
os.chdir(r"D:\gechengcheng3\Desktop\潜力类目挖掘")

# sale gmv_rate

# compete cr 系列 cr_rate系列


def log_x(df_cate, data_final_float_type):
    """
    对数变换缓解右偏性
    """
    for i in data_final_float_type:
        minumal = 0.0001  # 最小值为0时的附一个较低的值
        check = any(df_cate[i] <= 0)
        if check is True:
            abs_min_point = abs(min(df_cate[i]))
            df_cate[i] = df_cate[i].apply(lambda x: x+abs_min_point+minumal)
            df_cate[i] = df_cate[i].apply(lambda x: np.log(x))
        else: df_cate[i] = df_cate[i].apply(lambda x: np.log(x))
    return df_cate


data_zz = pd.read_csv(r"data_zz.txt", sep="\t", encoding="utf8")
# 对数变换处理右偏性
data_final_float_zz = list(data_zz.columns[12:])
data_zz = log_x(data_zz, data_final_float_zz)

data_jx = pd.read_csv(r"data_jx.txt", sep="\t", encoding="utf8")
# 对数变换处理右偏性
data_final_float_jx = list(data_jx.columns[11:])
data_jx = log_x(data_jx, data_final_float_jx)

"""
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
"""

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

month_compare = "2020-02"
# user-zz和jx对比图
fig, ax = plt.subplots(2, 2)
fig.suptitle('user_variance')
fig.subplots_adjust(hspace=0.3)
fontsize = 10
#
# jx_user.loc[(jx_user["item_first_cate_name"]==cate)&(jx_user["dt"] == month_compare), "user_pct"].skew()
# jx_user.loc[(jx_user["item_first_cate_name"]==cate)&(jx_user["dt"] == month_compare), "user_pct"].kurt()
#
# zz_user.loc[(zz_user["item_first_cate_name"]==cate)&(zz_user["dt"] == month_compare), "user_pct"].kurt()
# zz_user.loc[(zz_user["item_first_cate_name"]==cate)&(zz_user["dt"] == month_compare), "user_pct"].skew()

cate = "家庭清洁/纸品"
ax[0, 0].plot(jx_user.loc[(jx_user["item_first_cate_name"]==cate)&(jx_user["dt"] == month_compare), "user_pct"].values, label="jx_user_pct")
ax[0, 0].plot(zz_user.loc[(zz_user["item_first_cate_name"]==cate)&(zz_user["dt"] == month_compare), "user_pct"].values, "--",label="zz_user_pct")
ax[0, 0].legend()
ax[0, 0].set_title("user_pct", fontsize=fontsize)

ax[0, 1].plot(jx_user.loc[(jx_user["item_first_cate_name"]==cate)&(jx_user["dt"] == month_compare), "order_pct"].values, label="jx_order_pct")
ax[0, 1].plot(zz_user.loc[(zz_user["item_first_cate_name"]==cate)&(zz_user["dt"] == month_compare), "order_pct"].values, "--",label="zz_order_pct")
ax[0, 1].legend()
ax[0, 1].set_title("order_pct", fontsize=fontsize)

ax[1, 0].plot(jx_user.loc[(jx_user["item_first_cate_name"]==cate)&(jx_user["dt"] == month_compare), "order_rate"].values, label="jx_order_rate")
ax[1, 0].plot(zz_user.loc[(zz_user["item_first_cate_name"]==cate)&(zz_user["dt"] == month_compare), "order_rate"].values, "--",label="zz_order_rate")
ax[1, 0].legend()
ax[1, 0].set_title("order_rate", fontsize=fontsize)


ax[1,1].plot(jx_user.loc[(jx_user["item_first_cate_name"]==cate)&(jx_user["dt"] == month_compare), "user_rate"].values, label="jx_user_rate")
ax[1,1].plot(zz_user.loc[(zz_user["item_first_cate_name"]==cate)&(zz_user["dt"] == month_compare), "user_rate"].values, "--",label="zz_user_rate")
ax[1,1].legend()
ax[1,1].set_title("user_rate", fontsize=fontsize)


# traffic-zz和jx对比图
fig1, ax = plt.subplots(2, 2)
fig1.suptitle('traffic_variance')
fig1.subplots_adjust(hspace=0.3)
ax[0,0].plot(jx_traffic.loc[(jx_traffic["item_first_cate_name"]==cate)&(jx_traffic["dt"] == month_compare), "pv_pct"].values, label="jx_pv_pct")
ax[0,0].plot(zz_traffic.loc[(zz_traffic["item_first_cate_name"]==cate)&(zz_traffic["dt"] == month_compare), "pv_pct"].values, "--",label="zz_pv_pct")
ax[0,0].legend()
ax[0,0].set_title("pv_pct", fontsize=fontsize)

ax[0,1].plot(jx_traffic.loc[(jx_traffic["item_first_cate_name"]==cate)&(jx_traffic["dt"] == month_compare), "search_pct"].values, label="jx_search_pct")
ax[0,1].plot(zz_traffic.loc[(zz_traffic["item_first_cate_name"]==cate)&(zz_traffic["dt"] == month_compare), "search_pct"].values, "--",label="zz_search_pct")
ax[0,1].legend()
ax[0,1].set_title("search_pct", fontsize=fontsize)

ax[1,0].plot(jx_traffic.loc[(jx_traffic["item_first_cate_name"]==cate)&(jx_traffic["dt"] == month_compare), "pv_rate"].values, label="jx_pv_rate")
ax[1,0].plot(zz_traffic.loc[(zz_traffic["item_first_cate_name"]==cate)&(zz_traffic["dt"] == month_compare), "pv_rate"].values, "--",label="zz_pv_rate")
ax[1,0].legend()
ax[1,0].set_title("pv_rate", fontsize=fontsize)

ax[1,1].plot(jx_traffic.loc[(jx_traffic["item_first_cate_name"]==cate)&(jx_traffic["dt"] == month_compare), "search_rate"].values, label="jx_search_rate")
ax[1,1].plot(zz_traffic.loc[(zz_traffic["item_first_cate_name"]==cate)&(zz_traffic["dt"] == month_compare), "search_rate"].values, "--",label="zz_serach_rate")
ax[1,1].legend()
ax[1,1].set_title("search_rate")

# sale-zz和jx对比图
fig2, ax = plt.subplots(2, 2)
fig2.suptitle('sale_variance')
fig2.subplots_adjust(hspace=0.3)
ax[0,0].plot(jx_sale.loc[(jx_sale["item_first_cate_name"]==cate)&(jx_sale["dt"] == month_compare), "sale_cnt_pct"].values, label="jx_sale_cnt_pct")
ax[0,0].plot(zz_sale.loc[(zz_sale["item_first_cate_name"]==cate)&(zz_sale["dt"] == month_compare), "sale_cnt_pct"].values, "--",label="zz_sale_cnt_pct")
ax[0,0].legend()
ax[0,0].set_title("sale_cnt_pct")

ax[0,1].plot(jx_sale.loc[(jx_sale["item_first_cate_name"]==cate)&(jx_sale["dt"] == month_compare), "sale_cnt_rate"].values, label="jx_sale_cnt_rate")
ax[0,1].plot(zz_sale.loc[(zz_sale["item_first_cate_name"]==cate)&(zz_sale["dt"] == month_compare), "sale_cnt_rate"].values, "--", label="zz_sale_cnt_rate")
ax[0,1].legend()
ax[0,1].set_title("sale_cnt_rate", fontsize=fontsize)

ax[1,0].plot(jx_sale.loc[(jx_sale["item_first_cate_name"]==cate)&(jx_sale["dt"] == month_compare), "gmv_pct"].values, label="jx_gmv_pct")
ax[1,0].plot(zz_sale.loc[(zz_sale["item_first_cate_name"]==cate)&(zz_sale["dt"] == month_compare), "gmv_pct"].values, "--",label="zz_gmv_pct")
ax[1,0].legend()
ax[1,0].set_title("gmv_pct", fontsize=fontsize)

ax[1,1].plot(jx_sale.loc[(jx_sale["item_first_cate_name"]==cate)&(jx_sale["dt"] == month_compare), "gmv_rate"].values, label="jx_gmv_rate")
ax[1,1].plot(zz_sale.loc[(zz_sale["item_first_cate_name"]==cate)&(zz_sale["dt"] == month_compare), "gmv_rate"].values, "--", label="zz_gmv_rate")
ax[1,1].legend()
ax[1,1].set_title("gmv_rate", fontsize=fontsize)

# compete-zz和jx对比图
fig3, ax = plt.subplots(4, 3)
fig3.suptitle('compete_variance')
fig3.subplots_adjust(hspace=0.4)
ax[0,0].plot(jx_compete.loc[(jx_compete["item_first_cate_name"]==cate)&(jx_compete["dt"] == month_compare), "cr3"].values, label="jx_cr3")
ax[0,0].plot(zz_compete.loc[(zz_compete["item_first_cate_name"]==cate)&(zz_compete["dt"] == month_compare), "cr3"].values, "--",label="zz_cr3")
ax[0,0].legend()
ax[0,0].set_title("cr3", fontsize=fontsize)

ax[0,1].plot(jx_compete.loc[(jx_compete["item_first_cate_name"]==cate)&(jx_compete["dt"] == month_compare), "cr5"].values, label="jx_cr5")
ax[0,1].plot(zz_compete.loc[(zz_compete["item_first_cate_name"]==cate)&(zz_compete["dt"] == month_compare), "cr5"].values, "--",label="zz_cr5")
ax[0,1].legend()
ax[0,1].set_title("cr5", fontsize=fontsize)

ax[0,2].plot(jx_compete.loc[(jx_compete["item_first_cate_name"]==cate)&(jx_compete["dt"] == month_compare), "cr8"].values, label="jx_cr8")
ax[0,2].plot(zz_compete.loc[(zz_compete["item_first_cate_name"]==cate)&(zz_compete["dt"] == month_compare), "cr8"].values, "--",label="zz_cr8")
ax[0,2].legend()
ax[0,2].set_title("cr8", fontsize=fontsize)

ax[1,0].plot(jx_compete.loc[(jx_compete["item_first_cate_name"]==cate)&(jx_compete["dt"] == month_compare), "cr3_rate"].values, label="jx_cr3_rate")
ax[1,0].plot(zz_compete.loc[(zz_compete["item_first_cate_name"]==cate)&(zz_compete["dt"] == month_compare), "cr3_rate"].values, "--",label="zz_cr3_rate")
ax[1,0].legend()
ax[1,0].set_title("cr3_rate", fontsize=fontsize)

ax[1,1].plot(jx_compete.loc[(jx_compete["item_first_cate_name"]==cate)&(jx_compete["dt"] == month_compare), "cr5_rate"].values, label="jx_cr5_rate")
ax[1,1].plot(zz_compete.loc[(zz_compete["item_first_cate_name"]==cate)&(zz_compete["dt"] == month_compare), "cr5_rate"].values, "--",label="zz_cr5_rate")
ax[1,1].legend()
ax[1,1].set_title("cr5_rate", fontsize=fontsize)

ax[1,2].plot(jx_compete.loc[(jx_compete["item_first_cate_name"]==cate)&(jx_compete["dt"] == month_compare), "cr8_rate"].values, label="jx_cr8_rate")
ax[1,2].plot(zz_compete.loc[(zz_compete["item_first_cate_name"]==cate)&(zz_compete["dt"] == month_compare), "cr8_rate"].values, "--",label="zz_cr8_rate")
ax[1,2].legend()
ax[1,2].set_title("cr8_rate", fontsize=fontsize)

ax[2,0].plot(jx_compete.loc[(jx_compete["item_first_cate_name"]==cate)&(jx_compete["dt"] == month_compare), "top3_index"].values, label="jx_top3_index")
ax[2,0].plot(zz_compete.loc[(zz_compete["item_first_cate_name"]==cate)&(zz_compete["dt"] == month_compare), "top3_index"].values, "--",label="zz_top3_index")
ax[2,0].legend()
ax[2,0].set_title("top3_index", fontsize=fontsize)

ax[2,1].plot(jx_compete.loc[(jx_compete["item_first_cate_name"]==cate)&(jx_compete["dt"] == month_compare), "top5_index"].values, label="jx_top5_index")
ax[2,1].plot(zz_compete.loc[(zz_compete["item_first_cate_name"]==cate)&(zz_compete["dt"] == month_compare), "top5_index"].values, "--",label="zz_top5_index")
ax[2,1].legend()
ax[2,1].set_title("top5_index", fontsize=fontsize)

ax[2,2].plot(jx_compete.loc[(jx_compete["item_first_cate_name"]==cate)&(jx_compete["dt"] == month_compare), "top8_index"].values, label="jx_top8_index")
ax[2,2].plot(zz_compete.loc[(zz_compete["item_first_cate_name"]==cate)&(zz_compete["dt"] == month_compare), "top8_index"].values, "--",label="zz_top8_index")
ax[2,2].legend()
ax[2,2].set_title("top8_index", fontsize=fontsize)

ax[3,0].plot(jx_compete.loc[(jx_compete["item_first_cate_name"]==cate)&(jx_compete["dt"] == month_compare), "top10_brand_nums"].values, label="jx_top10_brand_nums")
ax[3,0].plot(zz_compete.loc[(zz_compete["item_first_cate_name"]==cate)&(zz_compete["dt"] == month_compare), "top10_brand_nums"].values, "--",label="zz_top10_brand_nums")
ax[3,0].legend()
ax[3,0].set_title("top10_brand_nums", fontsize=fontsize)

ax[3,1].plot(jx_compete.loc[(jx_compete["item_first_cate_name"]==cate)&(jx_compete["dt"] == month_compare), "top20_brand_nums"].values, label="jx_top20_brand_nums")
ax[3,1].plot(zz_compete.loc[(zz_compete["item_first_cate_name"]==cate)&(zz_compete["dt"] == month_compare), "top20_brand_nums"].values, "--",label="zz_top20_brand_nums")
ax[3,1].legend()
ax[3,1].set_title("top20_brand_nums", fontsize=fontsize)

ax[3,2].plot(jx_compete.loc[(jx_compete["item_first_cate_name"]==cate)&(jx_compete["dt"] == month_compare), "top50_brand_nums"].values, label="jx_top50_brand_nums")
ax[3,2].plot(zz_compete.loc[(zz_compete["item_first_cate_name"]==cate)&(zz_compete["dt"] == month_compare), "top50_brand_nums"].values, "--",label="zz_top50_brand_nums")
ax[3,2].legend()
ax[3,2].set_title("top50_brand_nums", fontsize=fontsize)

