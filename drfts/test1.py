import pandas as pd
from pyspark.sql import SparkSession
spark= SparkSession\
                .builder \
                .appName("dataFrame") \
                .getOrCreate()
# Loads data.
ll3=pd.DataFrame([[1,2],[3,4]],columns=['a','b'])

cc=ll3.values.tolist()

dd = list(ll3.columns)
# df=spark.createDataFrame(ll3)

# turn pandas.DataFrame to spark.dataFrame
spark_df = spark.createDataFrame(cc, dd)

print('spark.dataFram=', spark_df.show())

# turn spark.dataFrame to pandas.DataFrame
pandas_df = spark_df.toPandas()

print('pandas.DataFrame=', pandas_df)

import pandas as pd
import os
os.chdir(r"D:\gechengcheng3\Desktop\CatePotentialFinal\CatePotentialFinal\data")

data_df = pd.read_csv(r"test_data.csv", sep="\t")
traffic = spark.createDataFrame(data_df)

traffic.show(5)  # table格式

traffic.take(5)  # row格式

traffic.collect()  # 返回所有的数据记录行 一般不使用