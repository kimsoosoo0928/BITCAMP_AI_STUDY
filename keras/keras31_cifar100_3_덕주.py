import numpy as np
from tensorflow.keras.datasets import cifar100
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

# 1-1. 데이터 전처리

x_train = x_train.reshape(50000, 32 * 32 * 3)
x_test = x_test.reshape(10000, 32 * 32 * 3)

#scaler = MinMaxScaler()
scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()

x_train = scaler.fit_transform(x_train) # train에 한해서 fit과 transform을 한번에 가능하다.
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3) 
x_test = x_test.reshape(10000, 32, 32, 3) 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), padding='valid', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(128, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D()) 

model.add(Conv2D(128, (2,2),padding='valid', activation='relu'))               
model.add(Conv2D(128, (2,2),padding='same', activation='relu'))               
model.add(MaxPool2D())

model.add(Conv2D(64, (2,2),padding='valid', activation='relu'))
model.add(Conv2D(64, (2,2),padding='same', activation='relu'))                                                                
model.add(MaxPool2D())

model.add(Flatten())  
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'loss', patience=7, mode='min', verbose=1)

import time

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=5000, batch_size=256, callbacks=[es],validation_split=0.03, verbose=2)

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

# 1.
plt.subplot(2,1,1)
#! 2개의 plt 을 만드는데 (1,1) 번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2.
plt.subplot(2,1,2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()