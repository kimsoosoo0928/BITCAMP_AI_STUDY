import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, MaxPool2D, GlobalAveragePooling1D, LSTM, GlobalAveragePooling2D
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
from tensorflow.python.keras.layers.core import Dropout
import numpy as np
import time
import matplotlib.pyplot as plt

# 1. 데이터
# 이미 테스트데이터와 트레인 데이터가 분리되어있다.
x_train = np.load('./_save/_npy/k55_x_data_train_cifar100.npy')
x_test = np.load('./_save/_npy/k55_x_data_test_cifar100.npy')
y_train = np.load('./_save/_npy/k55_y_data_train_cifar100.npy')
y_test = np.load('./_save/_npy/k55_y_data_test_cifar100.npy')



# 전처리
x_train = x_train.reshape(50000, 32 * 32 * 3)
# 데이터의 내용물과 순서가 바뀌면 안된다.
x_test = x_test.reshape(10000, 32 * 32 * 3)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
# 데이터의 내용물과 순서가 바뀌면 안된다.
x_test = x_test.reshape(10000, 32, 32, 3)

# x_train = x_train.reshape(50000, 32, 32, 3)
# 데이터의 내용물과 순서가 바뀌면 안된다.
# x_test = x_test.reshape(10000, 32, 32, 3)

print(np.unique(y_train)) # [0 1 2 3 4 5 6 7 8 9]
print('y_shape : ', y_train.shape)
y_train = y_train.reshape(50000, 1)
y_test = y_test.reshape(10000, 1)
encoder = OneHotEncoder()
y_train = encoder.fit_transform(y_train)
y_train = np.c_[y_train.toarray()]
y_test = encoder.fit_transform(y_test)
y_test = np.c_[y_test.toarray()]

# 2. 모델링
# RNN
# model = Sequential()
# model.add(LSTM(8, input_shape=(32*32,3), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(100, activation='softmax'))



# DNN
# model = Sequential()
# model.add(Dense(528, input_shape=(32*32,3), activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(528, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(100, activation='softmax'))


# CNN
model = Sequential()
model.add(Conv2D(filters=128, activation='relu', kernel_size=(2,2), padding='valid',  input_shape=(32, 32, 3)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(Conv2D(128, (2,2), activation='relu', padding='valid'))
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(MaxPool2D())
model.add(GlobalAveragePooling2D())      
model.add(Dense(100, activation='softmax'))

# 3. 컴파일, 훈련       metrics['acc']
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(mode='auto', monitor='val_loss', patience=5)
cp = ModelCheckpoint(monitor='val_loss', mode='auto', filepath='./_save/ModelCheckPoint/keras48_9_MCP.hdf', save_best_only=True)
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=512, validation_split=0.05, verbose=1, callbacks=[es, cp])
# model.save('./_save/ModelCheckPoint/keras48_9_model.h5')
# model =load_model('./_save/ModelCheckPoint/keras48_9_model.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_9_MCP.hdf')

end = time.time() - start

print("걸린시간 : ", end)
# 4. 평가, 예측 predict X
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('acc : ', loss[1])