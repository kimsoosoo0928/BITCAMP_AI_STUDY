import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, Dropout, GlobalAveragePooling2D
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping
import time

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (150, 4) (150,)
print(y)

# 원핫인코딩 one-hot-encoding (150,) -> (150, 3)
# 0 -> [1, 0, 0]
# 1 -> [0, 1, 0]
# 2 -> [0, 0, 1]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y) # 원핫인코딩끝!
print(y[:5])
print(y.shape) # (150,3)


#1. 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7,test_size=0.3, shuffle=True, random_state=8)

#1-1. 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, StandardScaler,QuantileTransformer
scaler = QuantileTransformer() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(105, 4, 1, 1)
x_test = x_test.reshape(45, 4, 1, 1)

#2. 모델 구성
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
model.add(Dense(3, activation='softmax'))

#3. 컴파일 및 훈련 + EarlyStopping
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy']) # 이진 분류에 사용되는 binary_crossentropy, metrics는 결과에 반영은 안되고 보여주기만 한다.

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, 
            validation_split=0.2 ,callbacks=[es]) 

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss[0])
print('accuracy : ', loss[1])