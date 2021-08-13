import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 1. data 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28 * 28 * 1)   
x_test = x_test.reshape(10000, 28 * 28 * 1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28 * 28, 1)   
x_test = x_test.reshape(10000, 28 * 28, 1)

ohe = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe.fit(y_train)
y_train = ohe.transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()


# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

input1 = Input(shape=(28*28, 1))
dense = LSTM(units=20, activation='relu')(input1)
dense = Dense(128, activation='relu')(dense)
dense = Dense(64, activation='relu')(dense)
dense = Dense(32, activation='relu')(dense)
dense = Dense(16, activation='relu')(dense)
output1 = Dense(10, activation='softmax')(dense)

model = Model(inputs=input1, outputs=output1)

# 3. compile, fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2,
    validation_split=0.15, callbacks=[es])
end_time = time.time() - start_time

# 4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])  