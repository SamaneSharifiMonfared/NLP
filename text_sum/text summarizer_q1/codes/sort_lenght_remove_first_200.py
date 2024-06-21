import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
plt.style.use('ggplot')
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud_best_worst import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image


df_new = pd.read_csv("score_roberta_numeric.csv")
df = pd.DataFrame(df_new)
# print(df)

length = []
for  x in df.answer:
    length.append(len(x))

df['length'] = length

df_sorted = df.sort_values('length')

df_sorted = df_sorted[200:]

df_sorted.to_csv("df_sorted_by_length.csv")


print(0)