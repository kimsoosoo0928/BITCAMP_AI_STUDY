from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])
x_pred = [6]

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)


y_predict = model.predict(x)
print('x의 예측값 : ', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)

print("r2스코어 : ", r2)

# 과제 2
# R2를 0.9 올려라!!!
# 일요일 밤 12시까지 