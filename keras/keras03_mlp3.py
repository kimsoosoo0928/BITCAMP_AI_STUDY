from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)]) # (3x10)

x = np.transpose(x) # (10,3)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
                [10,9,8,7,6,5,4,3,2,1]]) # (3,10)
y = np.transpose(y) # (10,3)



#2. 모델구성
model = Sequential() # 순차적으로 내려가는 모델
model.add(Dense(11, input_dim=3)) 
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3)) # output_dim

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=10000, batch_size=1)
#4. 평가 예측

loss = model.evaluate(x, y) # loss 반환
print('loss : ', loss)


x_pred = np.array([[0, 21, 201]])
print(x_pred.shape)  #(1,3) 
result = model.predict(x_pred) #뒤의 컬럼의 개수만 맞으면 돌아간다.
print('1,1,10의 예측값 :', result)

plt.scatter(x, y)
plt.plot(x[:,0], y_predict, color='red')
plt.plot(x[:,1], y_predict, color='blue')
plt.plot(x[:,2], y_predict, color='green')
plt.show()

