import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk


# Read first answers



df = pd.read_csv("comments1_translated.csv")

df = df.set_axis(['index', 'answer'], axis=1)

df = df.dropna()

df['answer'] = df['answer'].str.lower()

df_most_freq = df['answer'].value_counts().head(20)

df = df[df.answer != "nothing"]
df = df[df.answer != "no"]
df = df[df.answer != "no comment"]
df = df[df.answer != "none"]
df = df[df.answer != "no idea"]
df = df[df.answer != "no opinion"]
df = df[df.answer != "already"]
df = df[df.answer != "i don't know"]
df = df[df.answer != "no comments"]
df = df[df.answer != "x"]
df = df[df.answer != "nil"]
df = df[df.answer != "not sure"]
df = df[df.answer != "not really"]
df = df[df.answer != "none."]
df = df[df.answer != "n.a."]
df = df[df.answer != "no."]
df = df[df.answer != "i have no idea"]
df = df[df.answer != "no suggestions"]
df = df[df.answer != "nothing."]
df = df[df.answer != "no comment."]
df = df[df.answer != "nothing i can think of"]
df = df[df.answer != "n/a"]
df = df[df.answer != "nothing to add"]
df = df[df.answer != "are not"]
df = df[df.answer != "n.v.t."]
df = df[df.answer != "nill"]
df = df[df.answer != "no comments."]
df = df[df.answer != "i don't think so"]
df = df[df.answer != "don't know"]
df = df[df.answer != "unsure"]
df = df[df.answer != "i have no idea."]
df = df[df.answer != "i have no idea? this vi?c"]
df = df[df.answer != "do not have"]
df = df[df.answer != "i do not know"]


df_word_tkn = nltk.word_tokenize(df['answer'][0])

df_word_tag = nltk.pos_tag(df_word_tkn)

ent = nltk.chunk.ne_chunk(df_word_tag)


from tqdm.notebook import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sia = SIA()

res = {}

for i ,row in enumerate(tqdm(df.answer , total = len(df))):
    score = sia.polarity_scores(row)
    res[row] = score['compound']

score_df = pd.DataFrame(res ,  index=[0]).T
score_df = score_df.set_axis(['score'], axis=1)



