# diabets을 함수형으로 구현하시오
# 서머리 확인 
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score

#1. 데이터 
datasets = load_diabetes()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=66, test_size=0.5)

print(x.shape, y.shape) # (442, 10) (442,)

print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

print(y[:30])
print(np.min(y), np.max(y))

#2. 모델구성

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(10,)) 
dense1 = Dense(128)(input1) 
dense2 = Dense(64)(dense1)
dense3 = Dense(32)(dense2)
dense4 = Dense(16)(dense3)
dense5 = Dense(8)(dense4)
dense6 = Dense(4)(dense5)
dense7 = Dense(2)(dense6)
output1 = Dense(1)(dense7)

model = Model(inputs=input1, outputs=output1)

model.summary()

#3. 컴파일, 훈련

model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=100, validation_split=0.2,  batch_size=1, verbose=2)

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)
