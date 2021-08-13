'''
커맨드창에서
cd \
cd study_603
cd _save
cd _graph
dir/w
tensorboard --logdir=.

웹을 키고
http://127.0.0.1:6006 or
http://localhost:6006/
'''


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,4,3,5,6,7,8,9,10])

#2. 모델 구성
model = Sequential()
model.add(Dense(9, input_dim=1)) #첫번째 hidden layer가 노드가 5개라면, input은 1개.
model.add(Dense(9)) # 순차적 모델이기 때문에 input dimension을 명시해주지 않아도 된다.
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # 컴퓨터가 이해하도록 컴파일

tb = TensorBoard(log_dir='./_save/_graph', histogram_freq=0,
                write_graph=True, write_images=True)

model.fit(x, y, epochs=100, batch_size=1, callbacks=[tb], validation_split=0.2) # 훈련시키기

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict(x)
print('8의 예측값 : ', result)
