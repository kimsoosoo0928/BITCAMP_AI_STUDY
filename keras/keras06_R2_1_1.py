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
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)


y_predict = model.predict(x)
print('x의 예측값 : ', y_predict)


 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

# 평가지표들은 대부분 test로 평가한다. 
# test의 y라벨값과 훈련을 시킨 y_predict이 필요한다. 


