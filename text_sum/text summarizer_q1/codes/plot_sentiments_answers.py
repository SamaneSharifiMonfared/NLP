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


df_new = pd.read_csv("df_sorted_by_length.csv")
df = pd.DataFrame(df_new)
# print(df)

# fig = px.histogram(df.head(20), x='answer' , y='sentiment', template='plotly_white', title='Answers by sentiment')
# fig.update_xaxes(categoryorder='total descending', title='total').update_yaxes(title='Answers by sentiment')
# fig.show()

# fig = px.histogram(df.head(100).tail(20), x='answer', template='plotly_white', title='Answers by sentiment'
#                    , color='sentiment', nbins=6, log_y=True, barmode='group')
# fig.update_xaxes(categoryorder='category descending', title='Answers').update_yaxes(title='Answers')
# fig.show()



fig = px.scatter(df.head(50), x='answer', y='sentiment', hover_name='length', text='answer', size='length', color='sentiment', size_max=45
                 , template='plotly_white', title='Answers similarity and frequency', labels={'answer': 'Avg. Length<BR>(answer)'}
                 , color_continuous_scale=px.colors.sequential.Sunsetdark)
fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.show()

print(0)