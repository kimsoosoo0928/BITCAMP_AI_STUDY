import numpy as np
x1 = np.array([range(100), range(301, 401), range(1,101)]) 
x2 = np.array([range(101, 201), range(411, 511), range(100,200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)
# y1 = np.array([range(1001, 1101)])
# y1 = np.transpose(y1)          #(100, 1)
y = np.array(range(1001, 1101))

print(x1.shape, x2.shape, y.shape) # (100, 3), (100, 3), (100,)

from sklearn.model_selection import train_test_split 
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y,
        train_size=0.7, test_size=0.3, shuffle=True, random_state=66)
# x1, x2, y와 같이 여러인자를 받을 수 있다. 

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape=(3,)) # 
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(5, activation='relu', name='dense3')(dense2)
output1 = Dense(11, name='output1')(dense3)

#2-2. 모델2
input2 = Input(shape=(3,))
dense11 = Dense(10, activation='relu',name='dense11')(input2)
dense12 = Dense(10, activation='relu',name='dense12')(dense11)
dense13 = Dense(10, activation='relu',name='dense13')(dense12)
dense14 = Dense(10, activation='relu',name='dense14')(dense13)
output2 = Dense(12, name='output2')(dense14)

# 모델 2개를 구성한다. 
# shape=(3,)
# 2개이상은 리스트로 받는다.


from tensorflow.keras.layers import concatenate, Concatenate
merge1 = concatenate([output1, output2]) # node의 개수가 더하기가 된다. 
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1)(merge3) # 최종 output

model = Model(inputs=[input1, input2], outputs=last_output) # 모델 정의

# concatenate

model.summary()

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y_train, epochs=10, batch_size=8, verbose=1)

#4. 평가, 예측
results = model.evaluate([x1_test, x2_test], y_test) # loss와 mae값을 출력한다.
# print(results)
print("loss : ", results[0])
print("metrics['mae'] : ", results[1])