import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
plt.style.use('ggplot')
import nltk


df_new = pd.read_csv("scores.csv")
df = pd.DataFrame(df_new)

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Run Roberta model
def polarity_score_roberta(sentense):
    encoded_text = tokenizer(sentense,return_tensors='pt')
    output = model(**encoded_text)
    score = output[0][0].detach().numpy()
    scores = softmax(score)
    scores_dict = {
    'roberta_neg':scores[0],
    'roberta_neu':scores[1],
    'roberta_pos':scores[2]
    }
    return scores_dict

df_answer = df['answer']

from tqdm.notebook import tqdm

roberta_result = {}

for i ,row in enumerate(tqdm(df_answer , total = len(df))):
    
    score = polarity_score_roberta(row)

    

    if(score['roberta_neg'] > score['roberta_neu']):
        if(score['roberta_neg'] > score['roberta_pos']):
            roberta_result[row] = -score['roberta_neg']
        else:
            roberta_result[row] = +score['roberta_pos']
    else:
        if(score['roberta_neu'] > score['roberta_pos']):
            roberta_result[row] = 0
        else:
            roberta_result[row] = + score['roberta_pos']


score_df = pd.DataFrame(roberta_result , index=[0]).T

score_df.to_csv("score_roberta_numeric.csv")
