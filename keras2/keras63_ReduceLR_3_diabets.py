# diabets
# LSTM

import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. data 
datasets = load_diabetes()

x = datasets.data # (442, 10) 
y = datasets.target # (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)

# 1-1. data preprocessing

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

input1 = Input(shape=(10, 1))
dense = LSTM(units=20, activation='relu')(input1)
dense = Dense(128, activation='relu')(dense)
dense = Dense(64, activation='relu')(dense)
dense = Dense(32, activation='relu')(dense)
dense = Dense(16, activation='relu')(dense)
output1 = Dense(1)(dense)

model = Model(inputs=input1, outputs=output1)

# 3. compile, fit
# model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
optimizer = Adam(lr=0.01)
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5)
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2,
    validation_split=0.15, callbacks=[es])
end_time = time.time() - start_time

# 4. evaluate, predict
y_predict = model.predict([x_test])
print('x의 예측값 : ', y_predict)

loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print('loss : ', loss)
r2 = r2_score(y_test, y_predict)
print('R^2 score : ', r2)

'''
LSTM
time :  67.64215731620789
loss :  5050.38330078125
R^2 score :  0.22182555457432607

LSTM + RL
time :  71.75014042854309
loss :  [4490.95849609375, 4490.95849609375]
R^2 score :  0.3080230560490833

'''