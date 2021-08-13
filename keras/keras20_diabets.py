# 실습 diabets
# 1. loss와 R2로 평가를 함
# MinMax와 Standard 결과를 명시

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target



x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7,test_size=0.3, shuffle=True, random_state=9) 

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(55, input_shape=(10, ), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(12, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=300, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

'''
-MinMaxScaler-
epochs=300, batch_size=32
r2 score :  0.6053326096394793
'''

'''
-StandardScaler-
epochs=300, batch_size=32
r2 score :  0.5798717386532026
'''

