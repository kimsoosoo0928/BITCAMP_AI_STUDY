from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import time 
import matplotlib.pyplot as plt

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data 
y = datasets.target 
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
train_size=0.7, shuffle=True, random_state=9)

print(x_train.shape, x_test.shape) 
print(y_train.shape, y_test.shape) \

# 데이터 전처리 
# 2차원 -> 4차원 

scaler = PowerTransformer()
scaler.fit_transform(x_train)
scaler.transform(x_test)

x_train = x_train.reshape(398, 30, 1, 1)
x_test = x_test.reshape(171, 30, 1, 1)

#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(1,1), 
                    padding='same', input_shape=(1, 1, 1), activation='relu'))
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
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 및 훈련
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy']) # 이진 분류에 사용되는 binary_crossentropy, metrics는 결과에 반영은 안되고 보여주기만 한다.

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=100, batch_size=8, 
            validation_split=0.2 ,callbacks=[es])

start_time = time.time()
model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=[es], validation_split=0.01,
shuffle=True, batch_size=10)
end_time = time.time() - start_time

#4. 평가 
loss = model.evaluate(x_test, y_test)
print("걸린시간: ", end_time)
print('loss: ', loss[0])
print('accuracy: ', loss[1])