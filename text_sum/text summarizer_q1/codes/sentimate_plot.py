import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk

df_new = pd.read_csv("scores.csv")
df = pd.DataFrame(df_new)
# print(df)

ax = sns.barplot(data=df , x = 'answer' , y = 'score')
ax.set_title('Sentiment analytics of answers to Q1')
plt.show()

