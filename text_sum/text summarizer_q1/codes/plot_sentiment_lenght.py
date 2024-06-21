import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
plt.style.use('ggplot')
import nltk
import plotly.express as px


df_new = pd.read_csv("score_roberta_numeric.csv")
df = pd.DataFrame(df_new)
# print(df)

length = []
for  x in df.answer:
    length.append(len(x))

df['length'] = length

df_sorted = df.sort_values('length')

df_sorted = df_sorted[200:]

# sns.pairplot(data= df_sorted ,
#              vars=['sentiment','length'],
#              hue= 'length',
#              palette='tab10')

# plt.show()


fig = px.histogram(df_sorted, x='sentiment', template='plotly_white', title='Sentiment of Answers based on length')
fig.show()

print(df_sorted)
