from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x = np.array([range(10)]) # (1x10)

x = np.transpose(x) # (10,1)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
                [10,9,8,7,6,5,4,3,2,1]]) # (3,10)
y = np.transpose(y) # (10,3)



#2. 모델구성
model = Sequential() # 순차적으로 내려가는 모델
model.add(Dense(11, input_dim=1)) 
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3)) # output_dim

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=10)
#4. 평가 예측

loss = model.evaluate(x, y) # loss 반환
print('loss : ', loss)


x_pred = np.array([[9]])
print(x_pred.shape) #(1,1)
result = model.predict(x_pred) 
print('10, 1.3, 1의 예측값 :', result)

#5. 그래프
y_predict = model.predict(x)

plt.scatter(x, y[:,0])
plt.scatter(x, y[:,1])
plt.scatter(x, y[:,2])

plt.plot(x, y_predict[:,0], color='red')
plt.plot(x, y_predict[:,1], color='blue')
plt.plot(x, y_predict[:,2], color='green')
plt.show()


"""
loss :  0.005722664762288332
(1, 1)
의 예측값 : [[10.044145   1.4949055  0.9886673]]
"""

"""
"""

