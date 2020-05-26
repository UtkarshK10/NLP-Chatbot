
import pickle
import numpy as np
with open('train_qa.txt','rb') as f:
    train_data=pickle.load(f)
    
with open('test_qa.txt','rb') as f:
    test_data=pickle.load(f)
    
all_data=train_data+test_data

vocab = set()

for story,question,answer in all_data:
    vocab=vocab.union(set(story))
    vocab=vocab.union(set(question))
    
vocab.add('yes')
vocab.add('no')

print(vocab)

vocab_len=len(vocab)+1

#longest story
all_story_lens=[len(data[0]) for data in all_data]
max_story_len=max(all_story_lens)
#longest question
max_question_len=max([len(data[1]) for data in all_data])

import tensorflow as tf
from tensorflow import keras

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

tokenizer=Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

tokenizer.word_index

train_story_text=[]
train_question_text=[]
train_answers=[]

for story,question,answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)
    train_answers.append(answer)
    
train_story_seq=tokenizer.texts_to_sequences(train_story_text)
print(train_story_seq)
    

def vectorize_stories(data,word_index=tokenizer.word_index,max_story_len=max_story_len,max_question_len=max_question_len):
    X=[]
    #question
    Xq=[]
    #o/p(yes/no)
    Y=[]
    
    for story,query,answer in data:
        #for each story
        x=[word_index[word.lower()] for word in story]
        xq=[word_index[word.lower()] for word in query]
        y=np.zeros(len(word_index)+1)
        y[word_index[answer]]=1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X,maxlen=max_story_len),pad_sequences(Xq,maxlen=max_question_len),np.array(Y))


inputs_train,queries_train,answers_train = vectorize_stories(train_data)       
        
inputs_test,queries_test,answers_test = vectorize_stories(test_data)        
    
    
from keras.models import Sequential,Model
from keras.layers.embeddings import Embedding
from keras.layers import Input,Activation,Dense,Permute,Dropout,add,dot,concatenate,LSTM

#Placeholder shape=(max_story_len,batch_size)
input_sequence=Input((max_story_len,))
question=Input((max_question_len,))

vocab_size=len(vocab)+1

#Input encoder M
input_encoder_m=Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,output_dim=64))
input_encoder_m.add(Dropout(0.3))
#(samples,story_max_len,embedding_dim)

#Input Encoder C
input_encoder_c=Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3))
#(samples,story_max_len,question_max_len)


#question layer
question_encoder=Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,output_dim=64,input_length=max_question_len))
question_encoder.add(Dropout(0.3))
#(samples,question_max_len,embedding_dim)

#Encoder(input) -->encoded
input_encoded_m=input_encoder_m(input_sequence)
input_encoded_c=input_encoder_c(input_sequence)
question_encoded=question_encoder(question)


match = dot([input_encoded_m,question_encoded],axes=(2,2))
match= Activation('softmax')(match)

response= add([match,input_encoded_c])
response = Permute((2,1))(response)

answer=concatenate([response,question_encoded])

answer= LSTM(32)(answer)

answer= Dropout(0.5)(answer)
answer= Dense(vocab_size)(answer)
#(samples,vocab_size)
#Yes/No

answer=Activation('softmax')(answer)
model=Model([input_sequence,question],answer)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#model.summary()

history=model.fit([inputs_train,queries_train],answers_train,batch_size=32,epochs=10,validation_data=([inputs_test,queries_test],answers_test))

import matplotlib.pyplot as plt
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()


pred_results= model.predict(([inputs_test,queries_test]))
val_max=np.argmax(pred_results[0])
for key,val in tokenizer.word_index.items():
    if val == val_max:
        k=key

#xreation own story' and quest
'''
my_story="John left the kitchen . Sandra dropped the football in garden ."
my_story.split()
my_question = "Is the football in the garden ?"
my_question.split()
mydata= [(my_story.split(),my_question.split(),'no')]
my_story, my_ques, my_ans = vectorize_stories(mydata)
pred_results= model.predict(([my_story,my_ques]))
val_max=np.argmax(pred_results[0])
for key,val in tokenizer.word_index.items():
    if val == val_max:
        k=key
print(k)
pred_results[0][val_max]
'''