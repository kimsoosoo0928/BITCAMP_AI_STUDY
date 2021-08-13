from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


#1. 데이터
x = np.array([1,2,3]) # 스칼라3개 벡터1개 차원1개
y = np.array([1,2,3])

#2. 모델
model = Sequential() # Sequential 모델은 각 레이어에 정확히 하나의 입력 텐서와 하나의 출력 텐서가 있는 일반 레이어 스택에 적합
model.add(Dense(1, input_dim=1)) # .add() 메소드를 통해서 레이어 추가-> w, b가 계산된다. 

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # lose를 mse방식으로 줄인다는 의미, 최적화는 'adam' 방식을 이용 

model.fit(x, y, epochs=1000, batch_size=1) # x, y를 훈련을 시키겠다는 의미, epoch는 훈련 횟수 의미, 전체훈련양은 동일하다 *훈련할때마다 가중치가 변경된다.

#4. 평가, 예측
loss = model.evaluate(x, y) # loss 반환
print('loss : ', loss)

result = model.predict([4]) 
print('4의 예측값 : ', result) 

#--------------------------------------------------------------------
y_predict = model.predict(x)

plt.scatter(x, y)
plt.plot(x, y_predict, color='red')
plt.show()
