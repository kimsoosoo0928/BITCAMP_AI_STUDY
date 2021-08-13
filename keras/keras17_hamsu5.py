# 함수형 모델의 레이명에 대한 고찰

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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(5,)) 
xx = Dense(3)(input1) 
xx = Dense(4)(xx)
xx = Dense(10)(xx)
output1 = Dense(2)(xx)

model = Model(inputs=input1, outputs=output1) # 함수형은 모델을 마지막에 선언해준다.

model.summary()

# model = Sequential()
# model.add(Dense(3, input_shape=(5,)))
# model.add(Dense(4))
# model.add(Dense(10))
# model.add(Dense(2))

# model.summary() # 항상 사용한다. 

#3. 컴파일, 훈련
#4. 평가 예측