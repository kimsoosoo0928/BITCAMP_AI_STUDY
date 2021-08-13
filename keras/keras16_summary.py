import numpy as np


#1. 데이터
x = np.array([range(100), range(301, 401), range(1, 101), 
              range(100), range(401,501)])
x = np.transpose(x)
print(x.shape)

y = np.array([range(711, 811), range(101, 201)])
y = np.transpose(y)
print(y.shape)

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_shape=(5,)))
model.add(Dense(4))
model.add(Dense(10))
model.add(Dense(2))

model.summary() # 항상 사용한다. 