import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten
from sklearn.metrics import r2_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.python.keras.engine.training import Model

# 1. data 
datasets = load_breast_cancer()

x = datasets.data # (569, 30) 
y = datasets.target # (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)

# 1-1. data preprocessing

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


#2. 모델
model = Sequential()
model.add(Conv1D(128, 2, activation='relu', input_shape=(30, 1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


# 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss',save_best_only=True, mode='auto',
                      filepath='./_save/ModelCheckPoint/keras48_cancer_MCP.hdf5')

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2,
    validation_split=0.15, callbacks=[es])
end_time = time.time() - start_time

model = load_model('./_save/ModelCheckPoint/keras48_cancer_MCP.hdf5')
# model = load_model('./_save/ModelCheckPoint/keras48_3_model.h5')
# model.save('./_save/ModelCheckPoint/keras48_3_model.h5')

# 평가, 예측
loss = model.evaluate(x_test, y_test)
print('걸린시간 : ', end_time)
print('loss : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

'''
저장 실행
걸린시간 :  11.716644763946533
loss :  0.026805737987160683
r2 score :  0.8836059551171418

model
걸린시간 :  19.643623113632202
loss :  0.030655061826109886
r2 score :  0.866891689726554

load_model
걸린시간 :  11.233904600143433
loss :  0.026805737987160683
r2 score :  0.8836059551171418

check point
# 에러!
'''