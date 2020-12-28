import pandas as pd
import os
os.chdir("D:\gechengcheng3\Downloads")
wine_df = pd.read_csv(r"wine_0814_gcc_copy_20200831165920.csv", sep="\t", encoding="utf-8", nrows=100)
wine_df.to_csv(r"123.csv")

from wordseg import WordSegment
doc = u'十四是十四四十是四十，十四不是四十，四十不是十四'
ws = WordSegment(doc, max_word_len=2, min_aggregation=1, min_entropy=0.5)
ws.segSentence(doc)