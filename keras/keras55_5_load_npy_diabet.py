import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Flatten, LSTM
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from tensorflow.python.keras.layers.core import Dropout

x_data = np.load('./_save/_npy/k55_x_data_wine.npy')
y_data = np.load('./_save/_npy/k55_y_data_wine.npy')

font_path = "C:/Windows/Fonts/gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# 완성하시오
# acc 0.8 이상 만들것
# print(dataset.DESCR)
# print(dataset.feature_names)
# print(np.unique(y))

y_data = to_categorical(y_data)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.7, shuffle=True, random_state=66)

# print(x_train)
# print(x_train.shape)

scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape)
# print(x_test.shape)
x_train = x_train.reshape(124, 13, 1)
x_test = x_test.reshape(54, 13, 1)
# 

model = Sequential()
model.add(LSTM(units=128, activation='relu', input_shape=(13, 1)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))


#
cp = ModelCheckpoint(monitor='val_loss', mode='auto', filepath='./_save/ModelCheckPoint/keras48_5_MCP.hdf', save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', patience=15)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=1, epochs=70, validation_split=0.05, callbacks=[es, cp])

# model.save('./_save/ModelCheckPoint/keras48_5_model.h5')
# model =load_model('./_save/ModelCheckPoint/keras48_5_model.h5')
# model = load_model('./_save/ModelCheckPoint/keras48_5_MCP.hdf')

#
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
