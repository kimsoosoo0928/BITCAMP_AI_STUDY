# LSTM

import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LSTM, Conv1D, Flatten, Reshape, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping
import time

# 1. data 
datasets = load_diabetes()

x = datasets.data # (442, 10) 
y = datasets.target # (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.2, shuffle=True, random_state=66)

# 1-1. data preprocessing

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 3dim
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

#Conv1D 모델
model.add(Conv1D(128, 2, activation='relu', input_shape=(10, 1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))


# 3. compile, fit
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

import time 
start_time = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=2, validation_split=0.2, callbacks=[es])
end_time = time.time() - start_time

# 4. evaluate, predict
loss = model.evaluate(x_test, y_test)

print('=' * 25)
print('걸린시간 : ', end_time)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)

'''
LSTM + Conv1D
걸린시간 :  57.76119112968445
loss :  3521.614501953125
r2 :  0.4573817712973631
'''
