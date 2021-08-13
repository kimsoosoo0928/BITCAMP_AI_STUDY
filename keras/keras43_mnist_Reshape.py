import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

# x_train = x_train.reshape(60000, 28, 28, 1) # (60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1) # (10000, 28, 28, 1)


# * 데이터 전처리
ohe = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe.fit(y_train)
y_train = ohe.transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Reshape
from tensorflow.python.keras.layers.core import Flatten

model = Sequential()
model.add(Dense(units=10, activation='relu', input_shape=(28, 28)))
model.add(Flatten()) # (N, 280)
model.add(Dense(784) ) # (N, 784)
model.add(Reshape((28, 28, 1))) # (N, 28, 28, 1) *위의 dense와 같아야 한다.
# ! layer 안에서 Reshape 해줄 수 있다.
model.add(Conv2D(64, (2,2)))
model.add(Conv2D(64, (2,2)))
model.add(Conv2D(64, (2,2)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'loss', patience=20, mode='min', verbose=1)

import time

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=10, batch_size=10, callbacks=[es], validation_split=0.1, verbose=2)

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린 시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
걸린 시간 :  320.22744035720825
loss :  0.12790682911872864
accuracy :  0.9617999792098999
'''