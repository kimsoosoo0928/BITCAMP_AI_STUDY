# 06_R2_2를 카피
# 함수형으로 리폼하시오.    
# 서머리로 확인

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred = [6]

#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(1,))
dense1 = Dense(5, activation='selu')(input1) 
dense2 = Dense(3, activation='selu')(dense1)
output1 = Dense(1)(dense2)

model = Model(inputs=input1, outputs=output1)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)


y_predict = model.predict(x)
print('x의 예측값 : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)

print("r2스코어 : ", r2)