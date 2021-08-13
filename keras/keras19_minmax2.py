import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

print(np.min(x), np.max(x)) # 0.0 711.0

# 데이터 전처리 
# x = x/711.
# x = x/np.max(x)
# x = (x - np.min(x) / np.max(x) - np.min(x)) # MinMaxScalar 정규화 -> (x - min) / (max - min
# y는 정규화를 하지 않는다. 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x_scale = scaler.transform(x) # 변환
print(np.min(x_scale), np.min(x_scale))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scale, y,
        train_size=0.7,test_size=0.3, shuffle=True, random_state=66) # x 대신 x_scale을 넣어주어야 한다. 

# 문제점! -> Train과 Test 

#2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss="mse", optimizer="adam")
model.fit(x_train, y_train, epochs=100, batch_size=8)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)
