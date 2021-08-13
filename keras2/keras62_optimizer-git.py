import numpy as np
from tensorflow.keras import optimizers
import tensorflow as tf

# 1. 데이터 
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

# 3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adagrad, Adamax, Adadelta
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# optimizer = Adam(lr=0.0001)
# loss :  5.01643160604609e-13 결과물 :  [[11.]]
# loss :  2.495383344403024e-10 결과물 :  [[10.999984]]
# loss :  0.000617280718870461 결과물 :  [[11.049436]]

# optimizer = Adagrad(lr=0.0001)
# loss :  0.6413053274154663 결과물 :  [[12.355656]]
# loss :  0.0001995942584471777 결과물 :  [[10.969769]]
# loss :  0.00035476055927574635 결과물 :  [[10.984546]]

# optimizer = Adamax(lr=0.0001)
# loss :  0.7122852206230164 결과물 :  [[9.885519]]
# loss :  4.110872282581113e-07 결과물 :  [[10.999562]]
# loss :  2.3817576220608316e-05 결과물 :  [[10.995906]]

# optimizer = Adadelta(lr=0.0001)
# loss :  0.0022878064773976803 결과물 :  [[10.911067]]
# loss :  0.000507168413605541 결과물 :  [[11.021556]]
# loss :  22.904010772705078 결과물 :  [[2.5048692]]

# optimizer = RMSprop(lr=0.01)
# loss :  464307.90625 결과물 :  [[1348.0039]]
# loss :  0.04436270147562027 결과물 :  [[11.391051]]
# loss :  0.0002612279204186052 결과물 :  [[10.969543]]

# optimizer = Nadam(lr=0.01)
# loss :  3.603872842511402e-12 결과물 :  [[11.000003]]
# loss :  0.0005858407821506262 결과물 :  [[11.042584]]
# loss :  4.368771837348628e-12 결과물 :  [[10.999999]]





model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=1)
y_pred = model.predict([11])

print('loss : ', loss, '결과물 : ', y_pred)

