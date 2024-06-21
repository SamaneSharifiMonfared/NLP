import numpy as np
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.feature_extraction.text import CountVectorizer
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from deep_translator import GoogleTranslator
import string




def main():


    csv1 = pd.read_csv("comments1.csv" , encoding='latin-1')

    csv2 = pd.read_csv("comments2.csv" , encoding='latin-1')

    csv3 = pd.read_csv("comments3.csv" , encoding='latin-1')

    csv4 = pd.read_csv("comments4.csv" , encoding='latin-1')

    csv5 = pd.read_csv("comments5.csv" , encoding='latin-1')

    df1 = pd.DataFrame(csv1)
    df2 = pd.DataFrame(csv2)
    df3 = pd.DataFrame(csv3)
    df4 = pd.DataFrame(csv4)
    df5 = pd.DataFrame(csv5)

    df1 = df1.rename(columns = {"When considering our strategy, what -if anything- do you feel is currently missing or underexposed that you believe should be incorporated?" : "answers"})
    df2 = df2.rename(columns = {"What changes, if any, would make you feel more rewarded and recognized in your job?" : "answers"})
    df3 = df3.rename(columns = {"What do you value most in our Stronger25 strategy?" : "answers"})
    df4 = df4.rename(columns = {"Being an inclusive employer is important for a global company like Royal HaskoningDHV. What ideas..." : "answers"})
    df5 = df5.rename(columns = {"What additional support, if any, would be beneficial to further enhance your mental health, well-being, or work-life balance?" : "answers"})


# When considering our strategy, what -if anything- do you feel is currently missing or underexposed that you believe should be incorporated?
# What changes, if any, would make you feel more rewarded and recognized in your job?
# What do you value most in our Stronger25 strategy?
# Being an inclusive employer is important for a global company like Royal HaskoningDHV. What ideas...
# What additional support, if any, would be beneficial to further enhance your mental health, well-being, or work-life balance?

    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()
    df4 = df4.dropna()
    df5 = df5.dropna()

    df1_translated = []
    df2_translated = []
    df3_translated = []
    df4_translated = []
    df5_translated = []

    for x in df5['answers']:
        y = GoogleTranslator(source='auto', target='en').translate(x)
        df5_translated.append(y)
   
    df5_translated_df = pd.DataFrame(df5_translated)
    df5_translated_df.to_csv("comments5_translated.csv")

    print("Tokenizing here.")

    stop_words = ["per","I", "me", "the", "what", "which", "having", "for", "with", "of", "about", "but", "if", "both", "each", "any", "a"] # https://gist.github.com/sebleier/554280
    stemmer = SnowballStemmer("english") # Choose a language
    custom_tokenizer = RegexpTokenizer(r'\w+')

    def manipulate_str(a):
        a = str(a)
        a = a.lower()    
        word_list = custom_tokenizer.tokenize(a)
        
        stemmed_words = list()
        for w in word_list:
            sw = stemmer.stem(w)
            
            if w not in stop_words and len(sw) > 2:
                stemmed_words.append(sw)
            
        return ' '.join( set(stemmed_words) )
    
    item_name_en_tokenized = df1.answers.apply(lambda x: manipulate_str(x))
    print(item_name_en_tokenized)



    vectorizer = CountVectorizer(min_df=5, max_features=9000, stop_words="english") # would be 6000+ (up to 12k)
    vectorizer.fit(item_name_en_tokenized)
    text_features = vectorizer.transform(item_name_en_tokenized)

    print(text_features.shape)
    # stuff are super sparse here
    print(text_features.sum(axis=1).mean())


    np.random.seed(0)

    comps = list(range(2,22,2))
    var_ratio = []
    for n_comp in comps:
        svd = TruncatedSVD(n_components=n_comp, n_iter=10, random_state=2022)
        svd.fit(text_features)
        exp_var_ratio = svd.explained_variance_ratio_.sum()
        var_ratio.append(exp_var_ratio)
        print(f"With {n_comp} components, explained var ratio is {exp_var_ratio}")


 
    components = [0, 3, 6, 11, comps[-1]-3]

    f, axs = plt.subplots(len(components),1, figsize=(12, 4*len(components)))

    for i, component_to_viz in enumerate(components): # inefficient
        c = pd.DataFrame()
        c["magnitude"] = svd.components_[component_to_viz] * 100
        c["text"] = [f"txt_{c}" for c in vectorizer.get_feature_names_out()]
        
        c.sort_values("magnitude", ascending=False).head(16).plot.bar(x="text",
                                                                    y="magnitude",
                                                                    title=f"component {component_to_viz}", ax=axs[i])

    plt.tight_layout()
    print(plt.show())

    transformed_text_features = svd.transform(text_features) * 100 # finding it easier to read.
    n_feats_txt = transformed_text_features.shape[1]
    print(f"Merging {n_feats_txt} into {df1}")
    col_names = [f"txt_{c}" for c in range(1,n_feats_txt+1)]

    if type(text_features) is not pd.DataFrame:
        text_features = pd.DataFrame(transformed_text_features, columns=col_names)
        text_features[col_names] = text_features[col_names].astype(np.float16)

    items = pd.concat([df1, text_features], axis=1)
    print(items)

    plot_df = df1.merge(items, on="answers")

    fig = px.histogram(plot_df, x="answers", )
    fig.show()
    print(fig.show())



    # outpout -> Ich möchte diesen Text übersetzen


    # from sklearn.model_selection import train_test_split

    # df_train, df_test= train_test_split(df1, random_state=1)


    # vect = CountVectorizer()
    # vect.fit(df_train)

    # # learn training data vocabulary, then use it to create a document-term matrix
    # df_train_dtm = vect.transform(df_train)

    # # equivalently: combine fit and transform into a single step
    # df_train_dtm = vect.fit_transform(df_train)

    # print(type(df_train_dtm), df_train_dtm.shape)

    # df_test_dtm = vect.transform(df_test)
    # print(type(df_test_dtm), df_test_dtm.shape)

    # from sklearn.feature_extraction.text import TfidfTransformer

    # tfidf_transformer = TfidfTransformer()
    # tfidf_transformer.fit(df_train_dtm)
    # tfidf_transformer.transform(df_train_dtm)

    # from sklearn.naive_bayes import MultinomialNB
    # nb = MultinomialNB()

    # nb.fit(df_train_dtm, df_train)

    # from sklearn import metrics

    # # make class predictions for X_test_dtm
    # df_pred_class = nb.predict(df_test_dtm)

    # # calculate accuracy of class predictions
    # print("=======Accuracy Score===========")
    # print(metrics.accuracy_score(df_test, df_pred_class))

    # # print the confusion matrix
    # print("=======Confision Matrix===========")
    # metrics.confusion_matrix(df_test, df_pred_class)





main()



# When considering our strategy, what -if anything- do you feel is currently missing or underexposed that you believe should be incorporated?
# What changes, if any, would make you feel more rewarded and recognized in your job?
# What do you value most in our Stronger25 strategy?
# Being an inclusive employer is important for a global company like Royal HaskoningDHV. What ideas...
# What additional support, if any, would be beneficial to further enhance your mental health, well-being, or work-life balance?