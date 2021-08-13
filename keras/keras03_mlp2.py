from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
                [10,9,8,7,6,5,4,3,2,1]]) # (3x10)
print(x.shape)
x = np.transpose(x) 
print(x.shape) # (10,3)

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
print(y.shape) # (10,) <-> (10,1)


#2. 모델구성
model = Sequential() # 순차적으로 내려가는 모델
model.add(Dense(10, input_dim=3)) 
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=1000, batch_size=1)
#4. 평가 예측

loss = model.evaluate(x, y) # loss 반환
print('loss : ', loss)


x_pred = np.array([[10, 1.3, 1]])
print(x_pred.shape)  #(1,3) 
result = model.predict(x_pred) #뒤의 컬럼의 개수만 맞으면 돌아간다.
print('x_pre의 예측값 :', result)

#------------------------homework-------------------------------
y_predict = model.predict(x)

plt.scatter(x[:,0], y)
plt.scatter(x[:,1], y)
plt.scatter(x[:,2], y)
plt.plot(x[:,0], y_predict, color='red')
plt.plot(x[:,1], y_predict, color='blue')
plt.plot(x[:,2], y_predict, color='green')
plt.show()


