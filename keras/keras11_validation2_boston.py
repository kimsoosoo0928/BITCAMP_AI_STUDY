# 보스턴 주택 가격을 완성할것

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(datasets.feature_names)
print(datasets.DESCR)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7,test_size=0.3, shuffle=True, random_state=66)

#2. 모델
model = Sequential()
model.add(Dense(9, input_dim=13))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss="mse", optimizer="adam", loss_weights=1)
model.fit(x_train, y_train, epochs=100, batch_size=1)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print('y_predict : ', y_predict)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)


'''
epochs=10000
loss :  16.257169723510742
y_predict :  [[10.606466 ]
r2 score :  0.803222711840581
'''

'''
epochs=100 val - o
loss :  39.777503967285156
y_predict :  [[29.284147 ]
r2 score :  0.608547815948211
'''

'''
epochs=100 val - x
loss :  29.798198699951172
y_predict :  [[ 5.9212666 ]
r2 score :  0.6393216886343243
'''


# B = 흑인의 비율
# input 13, output 1(506)
 
 # 완료하시오.