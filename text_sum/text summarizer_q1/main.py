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
import plotly.express as px
import spacy
# from spacy.lang.en.stop_words import Stop_words
from string import punctuation
from heapq import nlargest



df_new = pd.read_csv("df_sorted_by_length.csv")
df = pd.DataFrame(df_new)

stopwords = list(STOPWORDS)
nlp = spacy.load('en_core_web_sm')

all_answers = " " 
for s in df['answer']:
    all_answers = all_answers + " " + s


doc = nlp(all_answers)

tokens = [token.text for token in doc]

print(tokens)

punctuation  = punctuation + '\n'

word_frequencies = {}

for word in doc:
    if word.text not in stopwords:
        if word.text not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

max_freq = max(word_frequencies.values())
# normalizing it
for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word]/max_freq


sentece_tokens = [sent for sent in doc.sents]

# print(sentece_tokens)


sentence_score = {}

for sent in sentece_tokens:
    for word in sent:
        if word.text in word_frequencies.keys():
            if sent not in sentence_score.keys():
                sentence_score[sent] = word_frequencies[word.text]
            else:
                sentence_score[sent] += word_frequencies[word.text]


select_length = int(len(sentece_tokens)*0.3)

sum = nlargest(3 , sentence_score, key = sentence_score.get)
final_summary = [word.text for word in sum]
summary = ' '.join(final_summary)

print(summary)


wordcloud = WordCloud(width = 1000, height = 500,background_color ='white',min_font_size = 10).generate(str(summary))
plt.imshow(wordcloud)
plt.axis("off")

# store to file
plt.savefig("wordcloud.png", format="png")
plt.show()

print(0)
