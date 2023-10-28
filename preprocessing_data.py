import pandas as pd
import string
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk

mydata = pd.read_csv("Emotions.csv", encoding="ISO-8859-1", on_bad_lines='skip' )
mydata = mydata.dropna();


# punctuation removal
string.punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
mydata['Statement']= mydata['Statement'].apply(lambda x:remove_punctuation(x))


# lowering the text
mydata['Statement']= mydata['Statement'].apply(lambda x: x.lower())

# tokenization
def tokenization(text):
    tokens = nltk.word_tokenize(text)
    return tokens
mydata['Statement']= mydata['Statement'].apply(lambda x: tokenization(x))


stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output
mydata['Statement']= mydata['Statement'].apply(lambda x:remove_stopwords(x))


# removing digits
def remove_digits(text):
    output= [i for i in text if not i.isdigit()]
    return output
mydata['Statement'] = mydata['Statement'].apply(lambda x: remove_digits(x))


# stemming
porter_stemmer = PorterStemmer()
def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text
mydata['Statement']=mydata['Statement'].apply(lambda x: stemming(x))


# lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
mydata['Statement']=mydata['Statement'].apply(lambda x:lemmatizer(x))