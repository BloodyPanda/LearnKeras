from keras.layers import Merge           # 的的是的
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.text import *
from keras.preprocessing.sequence import skipgrams

vocab_size = 5000  # 字典大小
embed_size = 300   # 向量大小

word_model = Sequential()
word_model.add(Embedding(vocab_size, embed_size, embeddings_initializer="glorot_uniform", input_length=1))
word_model.add(Reshape((embed_size, )))
context_model = Sequential()
context_model.add(Embedding(vocab_size, embed_size, embeddings_initializer="glorot_uniform", input_length=1))
context_model.add(Reshape((embed_size, )))

model = Sequential()
model.add(Merge([word_model, context_model], mode="dot"))
model.add(Dense(1, init="glorot_uniform", activation="sigmoid"))
model.compile(loss="mean_squared_error", optimizer="adam")

# 导入需要分析的文本
text = "I love green eggs and ham ."

# 声明tokenizer并输入文本运行， 生成一个词token序列
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

# tokenizer创建了一个字典，它把每个单词映射到一个整型id，并让它在word_index属性中可用。
# 我们提取索引值并提取一个双向查找的表
word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}

# 把输入的词序列转换成ID列表并将其传给skipgrams函数。
# 接着我们把生成的56个skip_gram数据组(pair, label)中的前10个打印出来：
wids = [word2id[w] for w in text_to_word_sequence(text)]
pairs, labels = skipgrams(wids, len(word2id))
print(len(pairs), len(labels))
for i in range(10):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
        id2word[pairs[i][0]], pairs[i][0],
        id2word[pairs[i][1]], pairs[i][1],
        labels[i]))
