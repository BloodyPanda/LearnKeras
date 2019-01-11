import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
np.random.seed(2019)             # 重复性设置

# 网络和训练
NB_EPOCH = 250
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10           # 输出个数等于数字个数
# OPTIMIZER = SGD()         # SGD 优化器
OPTIMIZER = Adam()        # Adam 优化器
N_HIDDEN = 128            # 隐藏层中输出个数
VALIDATION_SPLIT = 0.2    # 训练集中用作验证集的比例
DROPOUT = 0.3             # dropout丢弃神经元的概率

# 数据：混合并划分训练集和测试集数据
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_trian是60000行28*28的书据，变形为60000*784
RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# 归一化
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train sample')
print(X_test.shape[0], 'test sample')

# 将类向量转换为二值类别矩阵（One-Hot编码）
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# 10个输出
# 最后是softmax激活函数
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))  # 加入的第一个隐藏层
model.add(Activation('relu'))                        # 激励函数为RELU
model.add(Dropout(DROPOUT))                          # 加在隐藏层上的dropout层
model.add(Dense(N_HIDDEN))                           # 加入的第二个隐藏层
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))                          # 加在隐藏层上的dropout层
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

# 编译模型
# loss: 损失函数; 这里选择多分类对数损失函数(categorical_crossentropy)
# optimizer: 优化器; 这里选择SGD()
# metrics: 性能评估,该性能的评估结果讲不会用于训练; 这里选择accuracy准确率
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# fit() 模型训练
# epochs 训练轮数
# batch_size 优化器进行权重更新前要观察的训练实例数
# verbose 日志显示
# validation_split 验证集划分
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                    verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

# 对模型训练的结果进行评估
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])