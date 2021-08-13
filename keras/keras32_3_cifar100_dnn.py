import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3)

# 1.1 데이터 전처리

# CNN shape
# x_train = x_train.reshape(50000, 32, 32, 3) 
# x_test = x_test.reshape(10000, 32, 32, 3)

# DNN shape
x_train = x_train.reshape(50000, 32 * 32 * 3)  
x_test = x_test.reshape(10000, 32 * 32 * 3)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()

x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test)

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
# model.add(Conv2D(filters=128, kernel_size=(2,2), padding='valid', activation='relu', input_shape=(32, 32, 3)))
# model.add(Dropout(0.2))
# model.add(Conv2D(128, (2,2),padding='same', activation='relu'))
# model.add(MaxPool2D()) 

# model.add(Conv2D(128, (2,2),padding='valid', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(128, (2,2),padding='same', activation='relu'))               
# model.add(MaxPool2D())

# model.add(Conv2D(64, (2,2),padding='valid', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2,2),padding='same', activation='relu'))                                                                
# model.add(MaxPool2D())

# model.add(GlobalAveragePooling2D())
# # model.add(Flatten())  
# # model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(128, activation='relu'))
# model.add(Dense(100, activation='softmax'))


# DNN 모델
model.add(Dense(2048, input_shape =(32 * 32 * 3, ), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'loss', patience=10, mode='min', verbose=1)

import time

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=5000, batch_size=150, callbacks=[es], validation_split=0.2, verbose=2)

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

