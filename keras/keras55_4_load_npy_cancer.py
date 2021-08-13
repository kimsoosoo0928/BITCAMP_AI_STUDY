import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

x_data = np.load('./_save/_npy/k55_x_data_breast_cancer.npy')
y_data = np.load('./_save/_npy/k55_y_data_breast_cancer.npy')

# np.save('./_save/_npy/k55_x_data.npy', arr=x_data)
# np.save('./_save/_npy/k55_y_data.npy', arr=y_data)

datasets = load_breast_cancer()
# 1. 데이터
# 데이터셋 정보 확인
print(datasets.DESCR)
print(datasets.feature_names)

y_data = to_categorical(y_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, random_state=66, shuffle=True)


# 2. 모델
# input = Input(shape=(30,))
# dense1 = Dense(128)(input)
# dense2 = Dense(64)(dense1)
# dense3 = Dense(64)(dense2)
# dense4 = Dense(32)(dense3)
# dense5 = Dense(16)(dense4)
# output = Dense(1, activation='sigmoid')(dense5)
# 마지막 레이어의 activation은 linear, sigmoid로 간다. 0, 1의 값을 받고 싶으면 무조건 sigmoid사용. loss는 binary_crossentropy

scaler = QuantileTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train = x_train.reshape(398, 30, 1, 1)
x_test = x_test.reshape(171, 30, 1, 1)

# 3. 컴파일, 훈련
model = Sequential()
model.add(Conv2D(filters=64, activation='relu', kernel_size=(1,1), padding='valid', input_shape=(30, 1, 1)))
model.add(Conv2D(32, kernel_size=(1,1), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16, kernel_size=(1,1), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# metrics에 들어간 건 결과에 반영되지 않고 보여주기만 한다.

es = EarlyStopping(mode='min', monitor='val_loss', patience=5)
cp = ModelCheckpoint(filepath='./_save/ModelCheckPoint/keras48_3_MCP.hdf', mode='auto', monitor='val_loss', save_best_only=True)
print(x_train.shape)
print(y_train.shape)
model.fit(x_train, y_train, batch_size=1, epochs=100, validation_split=0.1, callbacks=[es, cp])

# model.save('./_save/ModelCheckPoint/keras48_3_model.h5')
# model =load_model('./_save/ModelCheckPoint/keras48_3_model.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_3_MCP.hdf')

# 평가, 예측
loss = model.evaluate(x_test, y_test) # evaluate는 loss과 metrics도 반환한다. binary_crossentropy의 loss, accuracy의 loss
print('loss : ', loss[0])
print('accuracy : ', loss[1])