from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10]) 
y = np.array([1,2,4,3,5,7,9,3,8,12])

#2. 모델구성
model = Sequential() # 순차적으로 내려가는 모델
model.add(Dense(10, input_dim=1)) 
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=10, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y) # loss 반환
print('loss : ', loss)

#result = model.predict([11]) 
#print('11의 예측값 : ', result)

y_predict = model.predict(x)

plt.scatter(x,y)
plt.plot(x, y_predict, color='red')
plt.show()
