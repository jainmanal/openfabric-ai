import warnings
import nltk
import json
import random
import numpy as np
from tensorflow import keras
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

words=[]
classes=[]
document=[]
ignore_words=['?','!']
data_file = open('config/data.json', encoding="utf8").read()
intents = json.loads(data_file)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        document.append((w,intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

lemmatizer = WordNetLemmatizer()
words = [
    lemmatizer.lemmatize(each.lower()) 
    for each in words 
    if each not in ignore_words
]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

training = []
output_empty = [0] * len(classes)
for doc in document:
    bag = []
    pattern_words = doc[0]
    pattern_words = [
        lemmatizer.lemmatize(word.lower()) 
        for word in pattern_words 
    ]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])]=1
    training.append([bag, output_row])
random.shuffle(training)
training =np.array(training, dtype=object)

train_x = list(training[:,0])
train_y = list(training[:,1])

model = keras.models.Sequential()
model.add(keras.layers.Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(train_y[0]),activation='softmax'))

sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y),epochs=200, batch_size=5, verbose=1)
model.save('chatbot.h5',hist)


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words=[
        lemmatizer.lemmatize(word.lower()) 
        for word in sentence_words 
    ]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for each_word in sentence_words:
        for word_i,each in enumerate(words):
            if each==each_word:
                bag[word_i]=1
    return(np.array(bag))

def predict_class(sentence,model):
    class_arr = bow(sentence, words)
    res = model.predict(np.array([class_arr]))[0]
    error = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>error]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list=[]
    
    for r in results:
        return_list.append({"intent":classes[r[0]],"probability":str(r[1])})
    return return_list

def predictResponse(text,responses):
    match_list = list()
    for text in text.split():
        for keyword in responses:
            if text.lower() in keyword.lower():
                match_list.append(keyword)
    if match_list:
        return max(set(match_list), key = match_list.count)
    else:
        return random.choice(responses)

def getResponse(ints,intents_json,text):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result = predictResponse(text,i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text,model)
    response = getResponse(ints,intents,text)
    return response
