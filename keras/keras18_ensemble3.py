# 실습 x2를 주석처리해서 제거한 후 소스를 완성하시오.

import numpy as np
x1 = np.array([range(100), range(301, 401), range(1,101)]) 
# x2 = np.array([range(101, 201), range(411, 511), range(100,200)])
x1 = np.transpose(x1)
# x2 = np.transpose(x2)
# y1 = np.array([range(1001, 1101)])
# y1 = np.transpose(y1)          #(100, 1)
y1 = np.array(range(1001, 1101))
y2 = np.array(range(1901, 2001))
y1 = np.transpose(y1)


from sklearn.model_selection import train_test_split 
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2, test_size=0.3, random_state=8, shuffle=True)

print(x1_train.shape, x1_test.shape,
        y1_train.shape, y1_test.shape,
        y2_train.shape, y2_test.shape)


#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape=(3,)) # 
dense1 = Dense(64, activation='relu', name='dense1')(input1)
dense2 = Dense(32, activation='relu', name='dense2')(dense1)
dense3 = Dense(16, activation='relu', name='dense3')(dense2)
output1 = Dense(32, )(dense3)

#2-2. 모델2
modeldense1 = Dense(24)(output1)
modeldense2 = Dense(24)(modeldense1)
modeldense3 = Dense(24)(modeldense2)
modeldense4 = Dense(24)(modeldense3)
output21 = Dense(7)(modeldense4)

modeldense11 = Dense(24)(output1)
modeldense12 = Dense(24)(modeldense11)
modeldense13 = Dense(24)(modeldense12)
modeldense14 = Dense(24)(modeldense13)
output22 = Dense(7)(modeldense14)

last_output1 = Dense(1)(output21)
last_output2 = Dense(1)(output22)

model = Model(inputs=input1, outputs=[last_output1, last_output2])

# 모델 2개를 구성한다. 
# shape=(3,)
# 2개이상은 리스트로 받는다.








model.summary()

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam', metrics=['mae'])
model.fit(x1_train, [y1_train, y2_train], epochs=100, batch_size=8, verbose=1, validation_split=0.1)


#4. 평가, 예측
results = model.evaluate(x1_test, y1_test)

# print(results)
print("loss : ", results[0])
print("metrics['mae'] : ", results[1])
