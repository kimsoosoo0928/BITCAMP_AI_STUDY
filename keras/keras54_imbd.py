from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)

# 실습시작

print(x_train.shape, x_test.shape) 
print(y_train.shape, y_test.shape) 
#(25000,) (25000,)
#(25000,) (25000,)

print("최대길이 : ", max(len(i) for i in x_train)) 
print("평균길이 : ", sum(map(len, x_train)) / len(x_train)) 
# 최대길이 :  2494
# 평균길이 :  238.71364

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=500, padding='pre') # (25000, 100)
x_test = pad_sequences(x_test, maxlen=500, padding='pre') # (25000, 100)
print(x_train.shape, x_test.shape) 
print(type(x_train), type(x_train[0]))
print(x_train[1])

# y 확인
print('y : ',np.unique(y_train)) # [0 1]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape) # (25000, 2) (25000, 2)

#2. 모델구성
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, GRU

model = Sequential()
model.add(Embedding(10000, 100))
model.add(GRU(128, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

model.summary()
#3. compile, fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss',save_best_only=True, mode='auto',
                     filepath='./_save/ModelCheckPoint/keras54_imbd_MCP.hdf5')

import time
start_time = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=512, validation_split=0.2, verbose=1, callbacks=[es, cp])
end_time = time.time() - start_time

# model = load_model('./_save/ModelCheckPoint/keras54_imbd_MCP.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print("loss : ", loss[0])
print("acc : ", loss[1])

'''
time :  100.20227479934692
loss :  0.557786762714386
acc :  0.8361600041389465
'''
