# 보스턴을 함수형으로 구현하시오
# 서머리 확인 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7,test_size=0.3, shuffle=True, random_state=66)

#2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(13,)) 
dense1 = Dense(128, activation='selu')(input1) 
dense2 = Dense(64, activation='selu')(dense1)
dense3 = Dense(32, activation='selu')(dense2)
dense4 = Dense(16, activation='selu')(dense3)
dense5 = Dense(8, activation='selu')(dense4)
dense6 = Dense(4, activation='selu')(dense5)
output1 = Dense(1)(dense6)

model = Model(inputs=input1, outputs=output1)

model.summary()


#3. 컴파일, 훈련 
model.compile(loss="mse", optimizer="adam", loss_weights=1)
model.fit(x_train, y_train, epochs=1000, batch_size=32)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print('y_predict : ', y_predict)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)