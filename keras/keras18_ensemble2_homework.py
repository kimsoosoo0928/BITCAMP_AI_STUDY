import numpy as np
x1 = np.array([range(100), range(301, 401), range(1,101)]) 
x2 = np.array([range(101, 201), range(411, 511), range(100,200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)
# y1 = np.array([range(1001, 1101)])
# y1 = np.transpose(y1)          #(100, 1)
y1 = np.array(range(1001, 1101))
y2 = np.array(range(1901, 2001))


from sklearn.model_selection import train_test_split 
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, x2, y1, y2,
        train_size=0.7, test_size=0.3, shuffle=True, random_state=66)

print(x1_train.shape, x1_test.shape,
        x2_train.shape, x2_test.shape,
        y1_train.shape, y1_test.shape,
        y2_train.shape, y2_test.shape)


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
merge1 = Concatenate()([output1, output2])
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)
# last_output = Dense(1)(merge3)
output21 = Dense(7)(merge3)
last_output1 = Dense(1)(output21)

output22 = Dense(8)(merge3)
last_output2 = Dense(1)(output22)

model = Model(inputs=[input1, input2], outputs=[last_output1, last_output2]) # 모델 정의

# concatenate

model.summary()

#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=1000, batch_size=8, verbose=1)

#4. 평가, 예측
results = model.evaluate([x1_test, x2_test], [y1_test, y2_test]) # loss와 mae값을 출력한다.
# print(results)
print("loss : ", results[0])
print("metrics['mae'] : ", results[1])

'''
Epoch 1000/1000
9/9 [==============================] - 0s 11ms/step - loss: 5.3628 - dense_3_loss: 1.4031 - dense_5_loss: 3.9596 - dense_3_mae: 1.0978 - dense_5_mae: 1.8395
1/1 [==============================] - 0s 214ms/step - loss: 2.4550 - dense_3_loss: 0.5484 - dense_5_loss: 1.9066 - dense_3_mae: 0.6374 - dense_5_mae: 1.2062
loss :  2.4550442695617676
metrics['mae'] :  0.5484365820884705
'''