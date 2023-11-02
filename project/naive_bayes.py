from preprocessing_data import mydata
import nltk
import pandas as pd
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