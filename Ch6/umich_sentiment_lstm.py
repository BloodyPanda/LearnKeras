from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import os

data_train = pd.read_csv('./umich-si650-nlp/train.csv')
data_test = pd.read_csv('./umich-si650-nlp/test.csv')

maxlen = 0
word_freqs = collections.Counter()
num_recs = 0

for idx, row in data_train.iterrows():
    words = nltk.word_tokenize(row['sentence'].lower())
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freqs[word] += 1
    num_recs += 1

'''
print(len(word_freqs))  # 2072个独立的词
print(maxlen)           # 每个句子最多包含42个词
'''

MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40

# 建立词和索引相互的查询表，包含伪词PAD和UNK
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in
              enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k,v in word2index.items()}

X = np.empty((num_recs, ), dtype=list)
y = np.zeros((num_recs, ))
i = 0

for index, row in data_train.iterrows():
    words = nltk.word_tokenize(row['sentence'].lower())
    seqs = []
    for word in words:
        if word in word2index:
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    X[i] = seqs
    y[i] = int(row['label'])
    i += 1
# X = sequence.pad_sequences(X, padding='post', value=word2index['PAD'], maxlen=maxlen)
X = sequence.pad_sequences(X, padding='post', value=word2index['PAD'], maxlen=MAX_SENTENCE_LENGTH)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)


EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10

# 张量输入使用sigmoid作为激活函数的全连接层，它的输出大小为1，因而它的输出或者为0，或者为1
# 使用Adam和二分交叉熵损失函数来编译模型
# 超参数EMBEDDING_SIZE、HIDDEN_LAYER_SIZE、BATCH_SIZE、NUM_EPOCHS

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE,
                    input_length=MAX_SENTENCE_LENGTH))
model.add(SpatialDropout1D(Dropout(0.2)))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    validation_data=(Xtest, ytest))

plt.subplot(211)
plt.title("Accuracy")
plt.plot(history.history["acc"], color='g', label="Train")
plt.plot(history.history["val_acc"], color="b", label="Validation")
plt.legend(loc="best")

plt.tight_layout()
plt.show()

