# keras41_split2_LSTM 가져오기

# 1 ~ 100 까지의 데이터를 

# x                    y
# 1, 2, 3, 4, 5        6
# ...
# 95, 96, 97, 98, 99   100

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.python.keras.layers.core import Flatten

# 1. data

x_data = np.array(range(1, 101))
x_predict = np.array(range(96, 105))
#         x               y
# 96, 97, 98, 99, 100     ?
# ...
# 101, 102, 103, 104, 105 ?

# 예상 결과값 : 101, 102, 103, 104, 105, 106

size = 6

def split_x(dataset, size):
    aaa = [] 
    for i in range(len(dataset) - size + 1): 
        subset = dataset[i : (i + size)]
        aaa.append(subset) 
    return np.array(aaa) 

dataset = split_x(x_data, size)

print(dataset)

x = dataset[:, :5].reshape(95, 5, 1)
y = dataset[:, 5]


print('데이터를 어떻게 처리해줘야할지 모르겠다. ')


# print("x : \n", x)
# print("y : ", y)

# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Conv1D

model = Sequential()
# model.add(LSTM(64, input_shape=(5, 1)))
model.add(Conv1D(64, 2, input_shape=(5, 1)))
model.add(LSTM(64, return_sequences=True)) # LSTM 다음에 Conv 사용하는경우가 많다.
model.add(Conv1D(64,2 ))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))
model.summary()

# 3. compile, fit
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

import time 
start_time = time.time()
model.fit(x, y, epochs=100, batch_size=32, verbose=2,
    validation_split=0.15, callbacks=[es])
end_time = time.time() - start_time

# 4. evaluate, predict
y_predict = model.predict([x])
print('x의 예측값 : ', y_predict)

loss = model.evaluate(x,y)
print("time : ", end_time)
print('loss : ', loss)
r2 = r2_score(y, y_predict)
print('R^2 score : ', r2)
