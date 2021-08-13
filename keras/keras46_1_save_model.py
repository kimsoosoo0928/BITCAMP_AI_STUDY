# overfit를 극복하자
# 1. 전체 훈련 데이터가 많이 한다.
# 2. normailzation
# 3. dropout

import numpy as np
from tensorflow.keras.datasets import cifar100, mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#* 데이터 전처리

x_train = x_train.reshape(60000, 28 *28 *1)
x_test = x_test.reshape(10000, 28 *28 *1)

scaler = MinMaxScaler()
#scaler = StandardScaler()
#scaler = MaxAbsScaler()
#scaler = RobustScaler()
#scaler = QuantileTransformer()
#scaler = PowerTransformer()

# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
#! train에 한해서 fit 과 transform을 한번에 가능하다.
x_train = scaler.fit_transform(x_train) 

x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(np.unique(y_train))       # [0 1 2 3 4 5 6 7 8 9]

ohe = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe.fit(y_train)
y_train = ohe.transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()


# 2. 모델구성
from tensorflow.keras.models import Sequential, load_model  #! 저장한 모델을 불러올때 import 해야함
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout

""" model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), padding='valid', activation='relu', input_shape=(28, 28, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D()) 
model.add(Conv2D(64, (2,2),padding='valid', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2),padding='same', activation='relu'))                                                                
model.add(MaxPool2D())
model.add(Flatten())  
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax')) """

model = load_model('/study/_save/keras45_1_save_model.h5')
#! import 한 후 사용
#^ 모델만 저장한것으로 새로 훈련하는 것이다. (따라서 값은 달라진다.)

#model.summary()

# model.save('/study/_save/keras45_1_save_model.h5')


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(monitor= 'loss', patience=10, mode='min', verbose=1)

import time

start_time = time.time()

hist = model.fit(x_train, y_train, epochs=2, batch_size=600, callbacks=[es],validation_split=0.25, verbose=2)

end_time = time.time() - start_time

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린시간 : ', end_time)
print('loss : ', loss[0])
print('accuracy : ', loss[1])