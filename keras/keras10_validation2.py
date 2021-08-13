from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])
x_val = np.array([11,12,13])
y_val = np.array([11,12,13])

## 잘라서 만들어라

#x_train = np.array([1,2,3,4,5,6,7]) # 훈련용 데이터, 공부하는거
#y_train = np.array([1,2,3,4,5,6,7])
#x_test = np.array([8,9,10]) # 평가용 데이터, 평가하는거
#y_test = np.array([8,9,10])
#x_val = np.array([11,12,13])
#y_val = np.array([11,12,13])

# ex) 주가데이터 70일을 train으로 30일을 테스트로 잡을 수 있다


#2. 모델구성
model = Sequential() # 순차적으로 내려가는 모델
model.add(Dense(10, input_dim=1)) 
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

# verbose의 defalut = 1 이다.

#4. 평가 예측

loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)
y_predict = model.predict([11])


