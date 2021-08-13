from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 1. data
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                       index_col=None, header=0 ) # (4898, 12)

x = datasets.iloc[:,0:11] # (4898, 11)
y = datasets.iloc[:,[11]] # (4898, 10)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y)
y = one.transform(y).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.15, shuffle=True, random_state=24)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. model 구성
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Dropout, GlobalAveragePooling1D, MaxPool1D

# model = Sequential()
# model.add(Conv1D(filters=32, kernel_size=2, padding='same',                          
#                         activation='relu', input_shape=(11, 1))) 
# model.add(Dropout(0.1))
# model.add(Conv1D(32, 2, padding='same', activation='relu'))
# model.add(MaxPool1D())
# model.add(Conv1D(64, 2, padding='same', activation='relu'))
# model.add(Dropout(0.1))
# model.add(Conv1D(64, 2, padding='same', activation='relu'))
# model.add(MaxPool1D())
# model.add(Conv1D(128, 2, padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv1D(128, 2, padding='same', activation='relu'))
# model.add(MaxPool1D())
# model.add(Conv1D(256, 2, padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(Conv1D(256, 2, padding='same', activation='relu'))
# model.add(GlobalAveragePooling1D())
# model.add(Dense(7, activation="softmax"))

# # 3. 컴파일 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss',save_best_only=True, mode='auto',
#                      filepath='./_save/ModelCheckPoint/keras48_wine_MCP.hdf5')
# import time 
# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2,
#     validation_split=0.1, callbacks=[es, cp])
# end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras48_wine_model_save.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_wine_model_save.h5')
model = load_model('./_save/ModelCheckPoint/keras48_wine_MCP.hdf5')

# 4. 평가 예측

loss = model.evaluate(x_test, y_test)
# print("time : ", end_time)
print('loss : ', loss[0])
print('acc : ', loss[1])

'''
저장 실행
loss :  0.9899269342422485
acc :  0.5374149680137634

model
loss :  0.9878330826759338
acc :  0.5863945484161377

load_model
loss :  0.9899269342422485
acc :  0.5374149680137634

check point
loss :  0.9899269342422485
acc :  0.5374149680137634
'''