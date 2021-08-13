import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

#* 데이터 전처리

#CNN shape
# x_train = x_train.reshape(60000, 28, 28, 1) 
# x_test = x_test.reshape(10000, 28, 28, 1)

#DNN shape
x_train = x_train.reshape(60000, 28 * 28)   #! 6만 행 의 28 * 28 열 인 2차원 데이터로 만들어서 CNN 모델 말고 DNN 모델로 할 수 있다.
x_test = x_test.reshape(10000, 28 * 28)

# print(np.unique(y_train))       # [0 1 2 3 4 5 6 7 8 9]

ohe = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe.fit(y_train)
y_train = ohe.transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()

#scaler = MinMaxScaler()
scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 3차원 shape
x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D, LSTM, Dropout, Conv1D

model = Sequential()
# CNN 모델
# model.add(Conv2D(filters=40, kernel_size=(2,2), padding='same', input_shape=(28,28,1)))
# model.add(Conv2D(20, (2,2), activation='relu'))             
# model.add(Conv2D(5, (2,2), activation='relu'))               
# model.add(MaxPool2D())
# model.add(Conv2D(20, (2,2), activation='relu'))
# model.add(Conv2D(10, (2,2), activation='relu'))
# model.add(MaxPool2D())

# # model.add(GlobalAveragePooling2D())

# model.add(Flatten())                                        
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# DNN 모델
# model.add(Dense(100, activation='relu', input_shape=(28*28,)))
# model.add(Dense(60, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(10, activation='softmax'))

# LSTM 모델
# model.add(LSTM(units=2048, activation='relu', input_shape=(28,28)))
# model.add(Dropout(0.2))
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# Conv1D 모델
model.add(Conv1D(2048, 2, activation='relu', input_shape=(28,28)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'loss', patience=50, mode='min', verbose=1)

import time

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=10, batch_size=1, callbacks=[es], validation_split=0.1, verbose=2)

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

''' 

'''