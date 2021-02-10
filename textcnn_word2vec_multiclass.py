# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#!/usr/bin/env python
# coding: utf-8

# In[86]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import json
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[87]:

#a=input('path of the taining dataset with fields as title ')
#b=input('path of test dataset')
#classes=int(input('number of classes'))
data_train=pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')

data_test=pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')

# In[88]:


data_train=data_train[:3000]


# In[89]:

data_test=data_test[3000:4000]
#data_test=data_train[1450:]


# In[90]:


#data_test


# In[91]:


#data_train=data_train[:100]


# In[ ]:





# In[ ]:





# In[92]:


#(data_train['Category'].value_counts())


# In[93]:


data_train.rename(columns={'Text':'title','Score':'tag'},inplace=True)
data_test.rename(columns={'Text':'title','Score':'tag'},inplace=True)


# In[94]:


#print('training dataset',data_train)

#print('testining dataset',data_test)


# In[95]:


#data_train


# In[96]:


#applying sentence tokenizer
import nltk.data 
tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle') 
# Loading PunktSentenceTokenizer using English pickle file 
def make_sent_token(x):
    return tokenizer.tokenize(x) 
#converting each paragraph into separate sentences


# In[97]:


data_train['sentence_token']=data_train['title'].apply(lambda x: make_sent_token(x))


# In[98]:


data_test['sentence_token']=data_test['title'].apply(lambda x: make_sent_token(x))


# In[99]:


#data_train


# In[100]:


data_train['no_of_sentences']=data_train['sentence_token'].apply(lambda x:len(x))


# In[101]:


data_test['no_of_sentences']=data_test['sentence_token'].apply(lambda x:len(x))


# In[102]:


#max(data_train['no_of_sentences'])##no of rows in sentence matrix which is to be feed in model(max number of sentence in any paragraph)


# In[103]:


#len(data_train[data_train['no_of_sentences']==88]['title'])


# In[104]:


#max(data_test['no_of_sentences'])


# In[105]:


def max_length_of_sentence(x,y):
    sen=x
    nu=y
    #print(sen)
    ma=0
    if(nu>1):
        l=sen.split('.')
        #print(l)
        for i in range(len(l)):
            k=l[i].replace(',','')
            maxi=len(k.split())
            #print(maxi)
            if(maxi>ma):
                ma=maxi
        return ma
    else:
        return len(sen.split())
        
    


# In[106]:


data_train['max_words_in_sentence']=data_train.apply(lambda x: max_length_of_sentence(x.title,x.no_of_sentences),axis=1)


# In[107]:


data_test['max_words_in_sentence']=data_test.apply(lambda x: max_length_of_sentence(x.title,x.no_of_sentences),axis=1)


# In[108]:


#max(data_train['max_words_in_sentence'])## number of columns in the data to be feeded


# In[109]:


m=max(data_train['no_of_sentences'])
n=max(data_train['max_words_in_sentence'])

print(m,n)

#So each para will be converted to a m*n matrix   (where m is the number of sentence and n is number of words in each sentence)


# # Major part starts here ..... Now converting the paragraph into required matrix

# In[110]:


import re
import string 
from nltk import word_tokenize
from nltk.corpus import stopwords
def make_tokens(text):     ##Converting into single tokens in order to create the vocabulary
    return word_tokenize(text)


data_train['tokens']=data_train['title'].apply(lambda x: make_tokens(x))
data_test['tokens']=data_test['title'].apply(lambda x: make_tokens(x))


# In[111]:


#data_train['tokens']


# In[112]:


from gensim import models
word2vec_path = '../input/word2vec-google/GoogleNews-vectors-negative300.bin'
word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


# In[113]:


all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in data_train["tokens"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths))
para_max=max(training_sentence_lengths)


# In[114]:


#len(TRAINING_VOCAB)


# In[115]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), char_level=False)
tokenizer.fit_on_texts(data_train['title'])       # we assigned values 


# In[116]:


train_word_index = tokenizer.word_index


# In[117]:


#(train_word_index)


# In[118]:


def make_train_seq(x):
    return tokenizer.texts_to_sequences(x)
data_train['train_seq']=data_train['sentence_token'].apply(lambda x:make_train_seq(x) )
data_test['train_seq']=data_test['sentence_token'].apply(lambda x:make_train_seq(x) )


# In[119]:


#(data_train['train_seq'])   # here every para has been encoded


# In[ ]:





# In[120]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
def padding(x):    #now padding each sentence to a length of n...number of columns
    MAX_SENTENCE_LENGTH=n  #(no of columns)
    return pad_sequences(x,maxlen=MAX_SENTENCE_LENGTH,padding='post')

data_train['padded']=data_train['train_seq'].apply(lambda x:padding(x))
data_test['padded']=data_test['train_seq'].apply(lambda x:padding(x))


# In[121]:


#(data_train.padded[8])


# In[122]:


# In[113]:

EMBEDDING_DIM=300
train_embedding_weights = np.zeros((len(train_word_index)+1, 
 EMBEDDING_DIM))
for word,index in train_word_index.items():
 train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print(train_embedding_weights.shape)


# In[123]:


def make_full_para(x):     #92 cross 192 matrix of a paragraph.   (m*n)
    l=len(x)
    h=m-l    #no. of extra rows to be added
    z=[0]*h*n       #1D vector(#addding extra lines for zeroes as padding)
    z=np.reshape(z,(h,n))    #reshaping it to match the dimension of paragraph
    s=x.tolist()+z.tolist()
    return s 


# In[ ]:





# In[124]:


data_train['full_para']=data_train['padded'].apply(lambda x : make_full_para(x))
data_test['full_para']=data_test['padded'].apply(lambda x : make_full_para(x))


# In[125]:


#data_train.full_para


# In[126]:


def create_1d_para(x):
    l=[]
    for i in x:
        l+=i    #concatenating all the sentences in a para into a single 1 d arrray
    return l
        
    


# In[127]:


data_train['single_d_array']=data_train['full_para'].apply(lambda x: create_1d_para(x) )
data_test['single_d_array']=data_test['full_para'].apply(lambda x: create_1d_para(x) )


# In[128]:


train_cnn_data=np.array(data_train['single_d_array'].tolist())


# In[129]:


p=np.array(data_test['single_d_array'].tolist())


# In[130]:


#p


# In[131]:



test_cnn_data=p


# In[132]:


label_names=data_train['tag'].unique()
new=pd.get_dummies(data_train['tag'])
label_names.tolist()
y_train=new[label_names].values


# In[133]:


print(label_names)
classes=len(label_names)


# In[134]:


#y_train=data_train['tag'].values
#y_test=data_test['tag'].values


# In[135]:


#y_train


# In[136]:


#from __future__ import print_function
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.text import text_to_word_sequence
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten,Bidirectional,GRU,LSTM,SpatialDropout1D,Reshape
from tensorflow.keras.layers import Embedding,concatenate
from tensorflow.keras.layers import Conv2D, GlobalMaxPooling2D,MaxPool2D,MaxPool3D,GlobalAveragePooling2D,Conv3D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


# In[137]:


#m


# In[138]:


filter_sizes = [1,2,3,4]
num_filters = 32
embed_size=300
embedding_matrix=train_embedding_weights
max_features=len(train_word_index)+1
maxlen=m*n


def get_model():    
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.4)(x)
    x = Reshape((m, n, 300))(x)
    #print(x)
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], 2), 
                                                                                    activation='relu')(x)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[0], 3),
                                                                                    activation='relu')(x)
    
    
    
    conv_4 = Conv2D(num_filters, kernel_size=(filter_sizes[1], 2), 
                                                                                    activation='relu')(x)
    conv_5 = Conv2D(num_filters, kernel_size=(filter_sizes[1], 3), activation='relu')(x)
    
    
    
    
    maxpool_0 = MaxPool2D()(conv_0)
    maxpool_0=Flatten()(maxpool_0)
    maxpool_1 = MaxPool2D()(conv_1)
    maxpool_1=Flatten()(maxpool_1)
    #maxpool_2 = MaxPool2D()(conv_2)
    #maxpool_3 = MaxPool2D()(conv_3)
    
    maxpool_4 = MaxPool2D()(conv_4)
    maxpool_4=Flatten()(maxpool_4)
    maxpool_5 = MaxPool2D()(conv_5)
    maxpool_5=Flatten()(maxpool_5)
    #maxpool_6 = MaxPool2D()(conv_6)
    #maxpool_6=Flatten()(maxpool_6)
    #maxpool_7 = MaxPool2D()(conv_7)
   # maxpool_7=Flatten()(maxpool_7)
        
    w=concatenate([maxpool_4, maxpool_5],axis=1)
    w=Flatten()(w)
    z = concatenate([maxpool_0, maxpool_1],axis=1)
    
    
    z = Flatten()(z)
    z=concatenate([w,z],axis=1)
    z=Dense(units=128,activation="relu")(z)
    z = Dropout(0.5)(z)
        
    outp = Dense(units=classes, activation="softmax")(z)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model


# In[139]:


#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#filters=[16,32,48,64]
#neurons=[16,32,64,128,256]
#rate=[0.1,0.2,0.3,0.4,0.5]
#rates=[0.1,0.2,0.3,0.4,0.5]


# In[140]:


#param_grid = dict( neurons = neurons,filters=filters,rates=rates)


# In[141]:


#clf = KerasClassifier(build_fn= get_model, epochs=5, batch_size=32, verbose= 0)


# In[142]:


#from sklearn.model_selection import GridSearchCV
#model09 = GridSearchCV(estimator= clf, param_grid=param_grid, n_jobs=-1)


# In[ ]:





# In[ ]:





# In[143]:


model=get_model()


# In[145]:


print(model.summary())


# In[146]:



#define callbacks
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]
hist = model.fit(train_cnn_data, y_train,  epochs=10,callbacks=callbacks_list,validation_split=0.1 )


# In[147]:


#(train_cnn_data)


# In[148]:


pred=model.predict(test_cnn_data)


# In[150]:


#data_test


# In[149]:


#pred


# In[151]:


original_ans=data_test['tag'].tolist()


# In[153]:


output_class_pred=[]
y_test=pred
y_test=y_test.tolist()
for i in range(len(y_test)):
    m=max(y_test[i])
    output_class_pred.append(label_names[y_test[i].index(m)])


# In[154]:


#output_class_pred


# In[155]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[161]:


print(classification_report(original_ans,output_class_pred))


# In[157]:


print(confusion_matrix(original_ans,output_class_pred))


# In[148]:


#pred=model09.predict(test_cnn_data.tolist())
#y_test=pred
#y_test=y_test.tolist()
#output_class_pred=[]
#for i in range(len(y_test)):
    #if(y_test[i][0]<0.5):
        #output_class_pred.append(0)
    #else:
        #output_class_pred.append(1)
        
#original_ans=data_test['tag']
#original_ans=original_ans.tolist()


# In[158]:


#as its a fake news classifier , so identifying a fake class will be a TP()
'''from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
def check_metric(output_class_pred,original_ans):
    rightly_predicted=0
    TP=0
    for i in range(len(y_test)):
        if(original_ans[i]==output_class_pred[i]):
            rightly_predicted+=1
        
        
    print("Overall_acuracy:",rightly_predicted/len(output_class_pred))
    print('TP',TP)
    accuracy=rightly_predicted/len(y_test)
    print(classification_report(original_ans,output_class_pred))
    print(confusion_matrix(original_ans,output_class_pred))
    TN=confusion_matrix(original_ans,output_class_pred)[0][0]
    TP=confusion_matrix(original_ans,output_class_pred)[1][1]
    FP=confusion_matrix(original_ans,output_class_pred)[0][1]
    FN=confusion_matrix(original_ans,output_class_pred)[1][0]
    
    precision=TP/(TP+FP)
    recalll=TP/(FN+TP)
    F1=2*precision*recalll/(precision+recalll)
    sensiti=TP/(TP+FN)
    specifici=TN/(TN+FP)
    numerator=TP*TN - FP*FN
    
    denominator=np.sqrt((TP+FP)*(FN+TN)*(FP+TN)* (TP+FN))
    MCc=numerator/denominator
    G_mean1=np.sqrt(sensiti*precision)
    G_mean2=np.sqrt(sensiti*specifici)
    print('precision:' ,TP/(TP+FP))
    print('recall:',TP/(FN+TP))
    print("F1:",F1)
    print("Specificity:",TN/(TN+FP))
    print("Sensitivity ",TP/(TP+FN))
    print('G-mean1:',np.sqrt(sensiti*precision))
    print("G-mean2",np.sqrt(sensiti*specifici))
    print("MCC :",MCc)
    acc=[]
    pre=[]
    recall=[]
    f1=[]
    specificity=[]
    sensitivity=[]
    GMean1=[]
    Gmean2=[]
    MCC=[]
    tp=[]
    fp=[]
    fn=[]
    tn=[]
    acc.append(accuracy)
    pre.append(precision)
    recall.append(recalll)
    f1.append(F1)
    specificity.append(specifici)
    sensitivity.append(sensiti)
    GMean1.append(G_mean1)
    Gmean2.append(G_mean2)
    MCC.append(MCc)
    tp.append(TP)
    fp.append(FP)
    tn.append(TN)
    fn.append(FN)
    data={'accuracy_all':acc,"precision":pre,'recall':recall,'F1_score':f1,'specificity':specificity,'sensitivity':sensitivity,'Gmean1':GMean1,"Gmean2":Gmean2,"MCC":MCC,"TP":tp,"FP":fp,"TN":tn,"FN":fn}
    metric=pd.DataFrame(data)
    return metric
    '''
    
    


# In[150]:


#resi=check_metric(output_class_pred,original_ans)


# In[66]:


#resi.to_csv('results.csv', mode='w', index = False, header=resi.columns,columns=resi.columns)


# In[ ]:





# In[ ]:





# In[67]:


## now perparing training data for yoon kim model


# In[162]:


def create_single_line_para(x):
    l=[]
    for i in x:
        l+=i    #concatenating all the sentences in a para into a single 1 d arrray
    return l
        


# In[163]:


data_train['create_single_line_para']=data_train['train_seq'].apply(lambda x: create_single_line_para(x) )
data_test['create_single_line_para']=data_test['train_seq'].apply(lambda x: create_single_line_para(x) )


# In[164]:


#(data_train['create_single_line_para'])


# In[165]:


from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[166]:


yoon_kim_train_data=np.array(data_train['create_single_line_para'].tolist())
yoon_kim_train_data=pad_sequences(yoon_kim_train_data,maxlen=para_max,padding='post')


# In[167]:


yoon_kim_test_data=np.array(data_test['create_single_line_para'].tolist())
yoon_kim_test_data=pad_sequences(yoon_kim_test_data,maxlen=para_max,padding='post')


# In[ ]:





# In[ ]:





# In[168]:


#from __future__ import print_function
from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.text import text_to_word_sequence
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten,Bidirectional,GRU,LSTM
from tensorflow.keras.layers import Embedding,concatenate
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D,MaxPooling1D,GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


# In[170]:


#train_y=pd.get_dummies(y_train)


# In[171]:


#trains_y=train_y[[0,1]].values


# In[172]:


embed_size=300
embedding_matrix=train_embedding_weights
max_features=len(train_word_index)+1
maxlen=para_max 
max_sequence_length=para_max
MAX_SEQUENCE_LENGTH=para_max
EMBEDDING_DIM=300


#model3 yoon kim


# In[173]:


def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, trainable=True, extra_conv=False):
    
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embeddings],
                            input_length=max_sequence_length,
                            trainable=trainable)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # Yoon Kim model (https://arxiv.org/abs/1408.5882)
    convs = []
    filter_sizes = [3,4,5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=2)(l_conv)
        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)

    # add a 1D convnet with global maxpooling, instead of Yoon Kim model
    #conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
    #pool = MaxPooling1D(pool_size=2)(conv)

    #if extra_conv==True:
        #x = Dropout(0.01)(l_merge)  
    #else:
        # Original Yoon Kim model
        #x = Dropout(0.001)(pool)
    x = Flatten()(l_merge)
    
    x = Dropout(0.5)(x)
    # Finally, we feed the output into a Sigmoid layer.
    # The reason why sigmoid is used is because we are trying to achieve a binary classification(1,0) 
    # for each of the 6 labels, and the sigmoid function will squash the output between the bounds of 0 and 1.
    preds = Dense(units=classes, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model


# In[174]:


model1 = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, 
                 True)


# In[175]:


training_data=yoon_kim_train_data


# In[176]:


testing_data=yoon_kim_test_data


# In[177]:



#define callbacks
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]
hist = model1.fit(training_data, y_train,  epochs=10,callbacks=callbacks_list,batch_size=32,validation_split=0.1 )


# In[178]:


pred=model1.predict(testing_data)


# In[179]:


#pred


# In[180]:


output_class_pred=[]
y_test=pred
y_test=y_test.tolist()
for i in range(len(y_test)):
    m=max(y_test[i])
    output_class_pred.append(label_names[y_test[i].index(m)])


# In[181]:


print(classification_report(original_ans,output_class_pred))
print(confusion_matrix(original_ans,output_class_pred))

# In[ ]:





# In[ ]:





# In[ ]:





# In[157]:


'''pred=model1.predict(testing_data)
y_test=pred
y_test=y_test.tolist()
output_class_pred=[]
#output_class_pred=[]
for i in range(len(y_test)):
    o=max(y_test[i])
    if(y_test[i].index(o)==0):
        output_class_pred.append(0)
    else:
        output_class_pred.append(1)
        
        
original_ans=data_test['tag']
original_ans=original_ans.tolist()'''


# In[158]:


#as its a fake news classifier , so identifying a fake class will be a TP
'''def check_metric(output_class_pred,original_ans):
    rightly_predicted=0
    TP=0
    for i in range(len(y_test)):
        if(original_ans[i]==output_class_pred[i]):
            rightly_predicted+=1
        
        
    print("Overall_acuracy:",rightly_predicted/len(output_class_pred))
    print('TP',TP)
    accuracy=rightly_predicted/len(y_test)
    print(classification_report(original_ans,output_class_pred))
    print(confusion_matrix(original_ans,output_class_pred))
    TN=confusion_matrix(original_ans,output_class_pred)[0][0]
    TP=confusion_matrix(original_ans,output_class_pred)[1][1]
    FP=confusion_matrix(original_ans,output_class_pred)[0][1]
    FN=confusion_matrix(original_ans,output_class_pred)[1][0]
    
    precision=TP/(TP+FP)
    recalll=TP/(FN+TP)
    F1=2*precision*recalll/(precision+recalll)
    sensiti=TP/(TP+FN)
    specifici=TN/(TN+FP)
    numerator=TP*TN - FP*FN
    
    denominator=np.sqrt((TP+FP)*(FN+TN)*(FP+TN)* (TP+FN))
    MCc=numerator/denominator
    G_mean1=np.sqrt(sensiti*precision)
    G_mean2=np.sqrt(sensiti*specifici)
    print('precision:' ,TP/(TP+FP))
    print('recall:',TP/(FN+TP))
    print("F1:",F1)
    print("Specificity:",TN/(TN+FP))
    print("Sensitivity ",TP/(TP+FN))
    print('G-mean1:',np.sqrt(sensiti*precision))
    print("G-mean2",np.sqrt(sensiti*specifici))
    print("MCC :",MCc)
    acc=[]
    pre=[]
    recall=[]
    f1=[]
    specificity=[]
    sensitivity=[]
    GMean1=[]
    Gmean2=[]
    MCC=[]
    tp=[]
    fp=[]
    fn=[]
    tn=[]
    acc.append(accuracy)
    pre.append(precision)
    recall.append(recalll)
    f1.append(F1)
    specificity.append(specifici)
    sensitivity.append(sensiti)
    GMean1.append(G_mean1)
    Gmean2.append(G_mean2)
    MCC.append(MCc)
    tp.append(TP)
    fp.append(FP)
    tn.append(TN)
    fn.append(FN)
    data={'accuracy_all':acc,"precision":pre,'recall':recall,'F1_score':f1,'specificity':specificity,'sensitivity':sensitivity,'Gmean1':GMean1,"Gmean2":Gmean2,"MCC":MCC,"TP":tp,"FP":fp,"TN":tn,"FN":fn}
    metric=pd.DataFrame(data)
    return metric'''
    
    
    


# In[159]:


#resi=check_metric(output_class_pred,original_ans)


# In[86]:


#resi.to_csv('results.csv', mode='w', index = False, header=resi.columns,columns=resi.columns)


# In[ ]:




