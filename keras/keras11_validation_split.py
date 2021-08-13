from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=66, test_size=0.5)

## train_test_split로 만들어라 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)

#x_train = np.array([1,2,3,4,5,6,7]) # 훈련용 데이터, 공부하는거
#y_train = np.array([1,2,3,4,5,6,7])
#x_test = np.array([8,9,10]) # 평가용 데이터, 평가하는거
#y_test = np.array([8,9,10])
#x_val = np.array([11,12,13])
#y_val = np.array([11,12,13])



#2. 모델구성
model = Sequential() # 순차적으로 내려가는 모델
model.add(Dense(10, input_dim=1)) 
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_val, y_val))
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.3, shuffle=True)  

# verbose의 defalut = 1 이다.

#4. 평가 예측

loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)
y_predict = model.predict([11])

