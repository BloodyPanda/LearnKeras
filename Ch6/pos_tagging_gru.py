from keras.layers.core import Activation, Dense, Dropout, RepeatVector, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import collections
import nltk
import numpy as np
import os

DATA_DIR = "."
fedata = open(os.path.join(DATA_DIR, "treebank_sents.txt"), "w")
ffdata = open(os.path.join(DATA_DIR, "treebank_poss.txt"), "w")

sents = nltk.corpus.treebank.tagged_sents()
for sent in sents:
    words, poss = [], []
    for word, pos in sent:
        if pos == "-NONE-":
            continue
        words.append(word)
        poss.append(pos)
    fedata.write("{}\n".format(" ".join(words)))
    ffdata.write("{}\n".format(" ".join(poss)))

fedata.close()
ffdata.close()


def parse_sentences(filename):
    word_freqs = collections.Counter()
    num_recs, max_len = 0, 0
    with open(filename) as fin:
        for line in fin:
            words = line.lower().strip().split()
            for word in words:
                word_freqs[word] += 1
            if len(words) > max_len:
                max_len = len(words)
            num_recs += 1
    return word_freqs, max_len, num_recs


# 运行这段代码可知：
# Words: 10947  Max Seq Len: 249  Records Num: 3914
# Tags: 45  Max Seq Len: 249  Records Num: 3914
s_wordfreqs, s_maxlen, s_numrecs = parse_sentences("treebank_sents.txt")
t_wordfreqs, t_maxlen, t_numrecs = parse_sentences("treebank_poss.txt")
print("Words:", len(s_wordfreqs), " Max Seq Len:", s_maxlen, " Records Num:", s_numrecs)
print("Tags:", len(t_wordfreqs), " Max Seq Len:", t_maxlen, " Records Num:", t_numrecs)

MAX_SEQLEN = 250
S_MAX_FEATURES = 5000
T_MAX_FEATURES = 45
s_vocabsize = min(len(s_wordfreqs), S_MAX_FEATURES) + 2
s_word2index = {x[0]:i+2 for i, x in enumerate(s_wordfreqs.most_common(S_MAX_FEATURES))}
s_word2index["PAD"] = 0
s_word2index["UNK"] = 1
s_index2word = {v:k for k, v in s_word2index.items()}

t_vocabsize = T_MAX_FEATURES + 1
t_word2index = {x[0]:i+1 for i, x in
    enumerate(t_wordfreqs.most_common(T_MAX_FEATURES))}
t_word2index['PAD'] = 0
t_index2word = {v:k for k, v in t_word2index.items()}


def build_tensor(filename, numrecs, word2index, maxlen, make_categorical=False, num_classes=0):
    data = np.empty((numrecs, ), dtype=list)
    fin = open(filename, "r")

    i = 0
    for line in fin:
        wids = []
        for word in line.lower().strip().split():
            if word in word2index.keys():
                wids.append(word2index[word])
            else:
                wids.append(word2index['UNK'])
        '''
        if make_categorical:
            data[i] = np_utils.to_categorical(wids, num_classes=num_classes)
        else:
            data[i] = wids
        '''

        #
        if make_categorical:
            wids = np.array([wids])
            wids = sequence.pad_sequences(wids, maxlen = maxlen)
            data[i] = np.array(np_utils.to_categorical(wids, num_classes=num_classes))
        # 如果是构建X，直接用ID即可，因为后面会用Embedding层处理
        else:
            data[i] = wids
        #

        i += 1

    if (make_categorical):
        pdata = np.array([d.reshape((d.shape[1], d.shape[2])) for d in data])
    else:
        pdata = sequence.pad_sequences(data, maxlen=maxlen)

    fin.close()
    #pdata = sequence.pad_sequences(data, maxlen=maxlen)
    return pdata


X = build_tensor(os.path.join(DATA_DIR, "treebank_sents.txt"), s_numrecs, s_word2index, MAX_SEQLEN)
Y = build_tensor(os.path.join(DATA_DIR, "treebank_poss.txt"), t_numrecs, t_word2index, MAX_SEQLEN, True, t_vocabsize)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

print(X.shape)
print(Y.shape)


EMBED_SIZE = 128
HIDDEN_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 1

model = Sequential()
model.add(Embedding(s_vocabsize, EMBED_SIZE, input_length=MAX_SEQLEN))
model.add(SpatialDropout1D(0.2))
model.add(GRU(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(RepeatVector(MAX_SEQLEN))
model.add(GRU(HIDDEN_SIZE, return_sequences=True))
model.add(TimeDistributed(Dense(t_vocabsize)))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=[Xtest, Ytest])
score, acc = model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE)
print("Test score: %.3f, accuracy: %.3f" % (score, acc))
