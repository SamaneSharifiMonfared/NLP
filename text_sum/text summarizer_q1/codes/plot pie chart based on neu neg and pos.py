import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
plt.style.use('ggplot')
import nltk


df_new = pd.read_csv("score_roberta.csv")
df = pd.DataFrame(df_new)
# print(df)

length = []
for  x in df.answer:
    length.append(len(x))
print(max(length))
df['length'] = length
print(df)


df_sorted = df.sort_values('length')


# plot pie chart based on neu neg and pos

# df_neg = df[df['sentiment']=='neu']

# df_sorted = df_neg.sort_values('length')

# df_top_10 = df_sorted[200:]

# df_top_10 = df_top_10.head(10)

# df_most_freq_2 = df_top_10['answer'].value_counts().head(20)

# ax = df_most_freq_2.plot(kind = "pie", title="Most Frequent Neutral Answers",
#                                               figsize=(10 , 10))
# plt.show()

