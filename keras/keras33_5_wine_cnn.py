import numpy as np
import pandas as pd
from scipy.sparse import data
from sklearn import datasets
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense, Conv2D, Dropout, GlobalAveragePooling2D
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
import time

#1. 데이터
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';',
                        index_col=None, header=0)

datasets = datasets.to_numpy()

x = datasets[:, 0:11]
y = datasets[:, 11:]

one_hot_Encoder = OneHotEncoder()
one_hot_Encoder.fit(y)
y = one_hot_Encoder.transform(y).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y, 
test_size=0.7, random_state=9, shuffle=True)

scaler = QuantileTransformer()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

x_train = x_train.reshape(1469, 11, 1, 1)
x_test = x_test.reshape(3429, 11, 1, 1)



# 모델구성
model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(1,1), 
                    padding='same', input_shape=(1, 1, 1), activation='relu'))
# model.add(Dropout(0, 2)) # 20%의 드롭아웃의 효과를 낸다 
model.add(Dropout(0.2))
model.add(Conv2D(16, (1,1), padding='same', activation='relu'))   

model.add(Conv2D(64, (1,1),padding='valid', activation='relu'))  
model.add(Dropout(0.2))
model.add(Conv2D(64, (1,1), padding='same', activation='relu')) 


model.add(Conv2D(128, (1,1), padding='valid', activation='relu')) 
model.add(Dropout(0.2))
model.add(Conv2D(128, (1,1), padding='same', activation='relu')) 

# 여기까지가 convolutional layer 

model.add(GlobalAveragePooling2D())
model.add(Dense(7, activation='softmax'))

# 컴파일 및 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', 
metrics=['accuracy'])

es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.02,
callbacks=[es])

# 평가
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0])
print('accuracy: ', loss[1])