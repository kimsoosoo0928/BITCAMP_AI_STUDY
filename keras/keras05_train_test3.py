from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import random

#1. 데이터
x = np.array(range(100)) 
y = np.array(range(1,101)) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7,test_size=0.2, shuffle=True, random_state=66) # 70% 데이터를 train으로 준다. random_state를 설정해줌으로써 값을 고정시켜준다.
print(x_test)
print(y_test)

random.setstate





#2. 모델구성
model = Sequential() # 순차적으로 내려가는 모델
model.add(Dense(10, input_dim=1)) 
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=50, batch_size=1)

#4. 평가 예측

loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)
y_predict = model.predict([100])
print('100의 예측값 : ', y_predict)

