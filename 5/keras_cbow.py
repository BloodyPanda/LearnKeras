from keras.models import Sequential
from keras.layers.core import Dense, Lambda
from keras.layers.embeddings import Embedding
import keras.backend as K

vocab_size = 5000
embed_size = 300
window_size = 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embed_size, embeddings_initializer='glorot_uniform',
                    input_length=window_size))
model.add((Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,))))
model.add(Dense(vocab_size, kernel_initializer='glorot_uniform'))

model.compile(loss='categorical_crossentropy', optimizer="adam")