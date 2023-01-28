import pandas as pd

# importing data

mydata = pd.read_csv("D:/faks/3/2/ORI/Projekat/Emotions.csv", encoding="ISO-8859-1", on_bad_lines='skip' )
mydata = mydata.dropna();

# data - text preprocessing

# punctuation removal
import string
string.punctuation

# defining the function
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

# applying the function
mydata['Statement']= mydata['Statement'].apply(lambda x:remove_punctuation(x))

# lowering the text

mydata['Statement']= mydata['Statement'].apply(lambda x: x.lower())

# tokenization

# defining the function
import nltk
def tokenization(text):
    tokens = nltk.word_tokenize(text)
    return tokens

# applying the function
mydata['Statement']= mydata['Statement'].apply(lambda x: tokenization(x))


# stopwords present in the library
stopwords = nltk.corpus.stopwords.words('english')

# defining the function
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

# applying the function
mydata['Statement']= mydata['Statement'].apply(lambda x:remove_stopwords(x))

# removing digits

# defining the function
def remove_digits(text):
    output= [i for i in text if not i.isdigit()]
    return output

# applying the function
mydata['Statement'] = mydata['Statement'].apply(lambda x: remove_digits(x))

# stemming

# importing the Stemming function from nltk library
from nltk.stem.porter import PorterStemmer

# defining the object for stemming
porter_stemmer = PorterStemmer()

# defining the function
def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text

# applying the function
mydata['Statement']=mydata['Statement'].apply(lambda x: stemming(x))

# lemmatization

from nltk.stem import WordNetLemmatizer

# defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

# defining the function
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

# applying the function
mydata['Statement']=mydata['Statement'].apply(lambda x:lemmatizer(x))





# naive bayes

from nltk.stem.lancaster import LancasterStemmer
stemmer= LancasterStemmer()

mydata = mydata.dropna()
training_data = mydata.to_dict("r")

corpus = {}
corpus_fear = {}
corpus_anger = {}
corpus_joy = {}
class_w = {}
responses = list(set([a["Emotion"] for a in training_data]))

for r in responses:    
    class_w[r] = []

for data in training_data:
    for word in data["Statement"]:
        if word not in corpus:
            corpus[word] = 1
        else:
            corpus[word] += 1
        class_w[data["Emotion"]].extend([word])
        if(data["Emotion"] == 'fear'):
            if word in corpus_fear:
                corpus_fear[word] += 1
            else:
                corpus_fear[word] = 1
        elif(data["Emotion"] == 'anger'):
            if word in corpus_anger:
                corpus_anger[word] += 1
            else:
                corpus_anger[word] = 1
        else:
            if word in corpus_joy:
                corpus_joy[word] += 1
            else:
                corpus_joy[word] = 1

def calculate_score(user_input, class_name):
     score = 0
     for word in nltk.word_tokenize(user_input):
          if stemmer.stem(word.lower()) in class_w[class_name]:
              if class_name == 'joy':
                  score += (1 / corpus[stemmer.stem(word.lower())]) + 0.2
              else:
                  score += (1 / corpus[stemmer.stem(word.lower())])
                   
     return score

def translate(user_input):
    
    emotion = []
    calculated_score = []
    for r in class_w.keys():
        emotion.append(r)
    for r in class_w.keys():
        calculated_score.append(calculate_score(user_input, r))
    df = pd.DataFrame(list(zip(emotion, calculated_score)),columns = ["Emotion", "Score"])
    print(df)
    out1 = df.loc[df["Score"].idxmax()]
    out2 = out1["Emotion"]
    if out1["Score"]==0 or ((calculated_score[1] == calculated_score[2]) > calculated_score[0]) or ((calculated_score[2] == calculated_score[0]) > calculated_score[1]):
        print("Tell me more.")
    else:
        if(out2 == "anger" or out2 == "fear"):
            print("You don't sound okey. I think you should seek help.")
        else:
            print('You sound great!');   

print ("\n Hi! Tell me how you feel. \n")
while (True):
   user_input = input().lower()
   if user_input == "quit":
      break
   translate(user_input)