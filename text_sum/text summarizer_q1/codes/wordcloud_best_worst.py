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
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
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

best = df_sorted.query('sentiment > 0').sort_values('sentiment', ascending=False).head(10)
worst = df_sorted.query('sentiment < 0').sort_values('sentiment').head(10)

#object of WordNetLemmatizer
lm = WordNetLemmatizer()
def text_transformation(df_col):
    corpus = []
    for item in df_col:
        new_item = re.sub('[^a-zA-Z]',' ',str(item))
        new_item = new_item.lower()
        new_item = new_item.split()
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus
corpus = text_transformation(df_sorted['answer'])

# rcParams = []
# rcParams['figure.figsize'] = 20,8
word_cloud = ""
for row in corpus:
    for word in row:
        word_cloud+=" ".join(word)
wordcloud = WordCloud(width = 1000, height = 500,background_color ='white',min_font_size = 10).generate(word_cloud)
plt.imshow(wordcloud)
plt.axis("off")

# store to file
plt.savefig("wordcloud.png", format="png")
plt.show()


print(0)