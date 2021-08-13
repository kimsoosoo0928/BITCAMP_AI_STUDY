import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# 1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)

# 1.1 데이터 전처리

# CNN shape
x_train = x_train.reshape(60000, 28, 28, 1) 
x_test = x_test.reshape(10000, 28, 28, 1)

# DNN shape
# x_train = x_train.reshape(60000, 28 * 28)   #! 6만 행 의 28 * 28 열 인 2차원 데이터로 만들어서 CNN 모델 말고 DNN 모델로 할 수 있다.
# x_test = x_test.reshape(10000, 28 * 28)

ohe = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe.fit(y_train)
y_train = ohe.transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()

# 2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

model = Sequential()
# CNN 모델
# model.add(Conv2D(filters=20, kernel_size=(3,3), padding='same', input_shape=( 28, 28, 1)))
# model.add(Conv2D(30, (3,3),padding='same', activation='relu'))             
# model.add(Conv2D(40, (4,4), activation='relu'))               
# model.add(MaxPool2D())
# model.add(Conv2D(10, (2,2),padding='same', activation='relu'))
# model.add(Conv2D(5, (2,2), activation='relu'))
# model.add(MaxPool2D())                                                                
# model.add(Flatten())                                        
# model.add(Dense(1000, activation='relu'))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# DNN 모델
model.add(Dense(1024, input_shape =(28 *28, ), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'loss', patience=50, mode='min', verbose=1)

import time

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=5000, batch_size=150, callbacks=[es], validation_split=0.2, verbose=2)

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


