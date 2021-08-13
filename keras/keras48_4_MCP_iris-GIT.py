import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LSTM, Conv1D, Flatten, Reshape, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

# 1. data 
datasets = load_iris()

x = datasets.data # (150, 4) 
y = datasets.target # (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)

# 1-1. data preprocessing

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Conv1D

model = Sequential()

# #DNN 모델
# model.add(Dense(150, activation='relu', input_dim = 10))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(1, activation='relu'))

# #CNN 모델
# model.add(Conv2D(filters=32, kernel_size=(2,1), padding='same', activation='relu', input_shape=(10,1, 1)))
# model.add(Conv2D(64, (2,1), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(128, (2,1) , padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (2,1), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,1), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (2,1), padding='same', activation='relu'))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(1))

# #LSTM모델
# model.add(LSTM(units=256, activation='relu', input_shape=(10,1)))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))

# #Conv1D 모델
# model.add(Conv1D(128, 2, activation='relu', input_shape=(4, 1)))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))

# 3. compile, fit
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
cp = ModelCheckpoint(monitor='val_loss', mode='auto', filepath='./_save/ModelCheckPoint/keras48_4_MCP.hdf', save_best_only=True)
model.compile(loss="mse", optimizer="adam", loss_weights=1)

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2,
    validation_split=0.15, callbacks=[es])
end_time = time.time() - start_time

model = load_model('./_save/ModelCheckPoint/keras48_4_MCP.hdf')
# model = load_model('./_save/ModelCheckPoint/keras48_4_model.h5')
# model.save('./_save/ModelCheckPoint/keras48_4_model.h5')

# 4. evaluate, predict
loss = model.evaluate(x_test, y_test)

print('=' * 25)
print('걸린시간 : ', end_time)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

'''
저장 실행
걸린시간 :  5.883350372314453
loss :  0.05629904195666313
r2 :  0.9149846662919001

model
걸린시간 :  7.139857530593872
loss :  0.05504513159394264
r2 :  0.9168781610671549

load_model
걸린시간 :  2.2087466716766357
loss :  0.05629904195666313
r2 :  0.9149846662919001

check point
에러!
'''