import nltk
import numpy as np

sents=nltk.corpus.treebank.tagged_sents()

fedata=open('treebank_sents.txt','w')
ffdata=open('treebank_poss.txt','w')
for sent in sents:
    words,poss=[],[]
    for word,pos in sent:
        if (pos=='-NONE-'):
            continue
        words.append(word)
        poss.append(pos)
    fedata.write("{}\n".format(" ".join(words)))
    ffdata.write("{}\n".format(" ".join(poss)))
fedata.close()
ffdata.close()

import collections
def parse_sentences(filename):
    word_freqs=collections.Counter()
    num_recs,max_len=0,0
    with open(filename) as f:
        for l in f:
            words=l.lower().strip().split()
            for w in words:
                word_freqs[w]+=1
            if(len(words)>max_len):
                max_len=len(words)
            num_recs+=1
    return word_freqs,max_len,num_recs

s_freq,s_maxlen,s_num=parse_sentences("treebank_sents.txt")
t_freq,t_maxlen,t_num=parse_sentences("treebank_poss.txt")
print("Words:",len(s_freq)," Max Seq Len:",s_maxlen," Records Num:",s_num)
print("Words:",len(t_freq)," Max Seq Len:",t_maxlen," Records Num:",t_num)

MAX_SEQLEN=100
S_FEATURES=5000
T_FEATURES=45
s_vocabsize=S_FEATURES+2
s_word2index={w[0]:i+2 for i,w in enumerate(s_freq.most_common(S_FEATURES))}
s_word2index['PAD']=0
s_word2index['UNK']=1
s_index2word={v:k for k,v in s_word2index.items()}

t_vocabsize=T_FEATURES+1
# 原书籍中这里有错
t_word2index={w[0]:i+1 for i,w in enumerate(t_freq.most_common(T_FEATURES))}
t_word2index['PAD']=0
t_index2word={v:k for k,v in t_word2index.items()}

from keras.utils import to_categorical
from keras.preprocessing import sequence
def build_tensor(filename,num_recs,word2index,max_len,
                 make_categorical=False,num_classes=0):
    data=np.empty((num_recs,),dtype=list)
    fin=open(filename,'r')
    for i,line in enumerate(fin):
        wids=[]
        words=line.lower().strip().split()
        for w in words:
            if(w in word2index.keys()):
                wids.append(word2index[w])
            else:
                wids.append(word2index['UNK'])
        # 如果是构建Y，需要用one-hot编码
        if make_categorical:
            wids=np.array([wids])
            wids=sequence.pad_sequences(wids,maxlen=max_len)
            data[i]=np.array(to_categorical(wids,num_classes=num_classes))
        # 如果是构建X，直接用ID即可，因为后面会用Embedding层处理
        else:
            data[i]=wids
    if(make_categorical):
        pdata=np.array([d.reshape((d.shape[1],d.shape[2])) for d in data])
    else:
        pdata=sequence.pad_sequences(data,maxlen=max_len)
    fin.close()
    return pdata
X=build_tensor('treebank_sents.txt',s_num,s_word2index,MAX_SEQLEN)
Y=build_tensor('treebank_poss.txt',t_num,t_word2index,MAX_SEQLEN,
              make_categorical=True,num_classes=t_vocabsize)

from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2,random_state=42)

from keras import Sequential
from keras.layers import Embedding,SpatialDropout1D,GRU,LSTM,RepeatVector,TimeDistributed,Activation
from keras.layers import Dense,TimeDistributed
from keras.activations import softmax
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
EMBED_SIZE=128
HIDDEN_SIZE=128
BATCH_SIZE=32
model=Sequential()
model.add(Embedding(s_vocabsize,EMBED_SIZE,
                   input_length=MAX_SEQLEN))
model.add(SpatialDropout1D(0.2))
model.add(GRU(HIDDEN_SIZE,dropout=0.2,recurrent_dropout=0.2))
model.add(RepeatVector(MAX_SEQLEN))
model.add(GRU(HIDDEN_SIZE,return_sequences=True))
model.add(TimeDistributed(Dense(t_vocabsize)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',
              metrics=['accuracy'])

NUM_EPOCHS=1
model.fit(Xtrain,Ytrain,batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,
         validation_data=[Xtest,Ytest])
score,acc=model.evaluate(Xtest,Ytest,batch_size=BATCH_SIZE)
print('Test score:%.3f,accuracy:%.3f'%(score,acc))
