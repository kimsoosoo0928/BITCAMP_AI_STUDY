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

model = Sequential()
model.add(Dense(100, input_dim=10, activation='relu'))
model.add(Dense(50, activation='relu')) 
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='relu'))# 활성화함수의 디폴트값이 있다.
model.add(Dense(1))
 # 마지막 레이어에 활성화함수를 쓰지 않는다.

#3. 컴파일, 훈련

model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=1000, validation_split=0.2,  batch_size=1, verbose=2)

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

# mse, R2

# 과제 1.
# 0.62 까지 올릴것!!

'''
model = Sequential()
model.add(Dense(100, input_dim=10, activation='relu'))
model.add(Dense(70, activation='relu')) 
model.add(Dense(50, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
epochs=10, validation_split=0.2,  batch_size=1
r2 score :  0.46081133017628084
'''

'''
model = Sequential()
model.add(Dense(100, input_dim=10, activation='relu'))
model.add(Dense(50, activation='relu')) 
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Dense(1))
epochs=100, batch_size=1
r2 score :  0.461552699146519
'''

'''
model = Sequential()
model.add(Dense(100, input_dim=10, activation='relu'))
model.add(Dense(50, activation='relu')) 
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='relu'))
model.add(Dense(1))
epochs=1000, batch_size=1
loss :  5555.11669921875
r2 score :  0.08093484703570464
'''