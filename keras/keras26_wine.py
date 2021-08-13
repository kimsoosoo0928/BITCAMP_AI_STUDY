import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names) 

x = datasets.data
y = datasets.target

print(x.shape, y.shape) 

# 완성하시오 !!!
# acc 0.8 이상 만들것 !!!

# 1-1 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7,test_size=0.3, shuffle=True, random_state=8)

# 1-2 데이터 전처리
from sklearn.preprocessing import QuantileTransformer
scaler = QuantileTransformer() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='softmax'))

#3. 컴파일 및 훈련 + EarlyStopping
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy']) # 이진 분류에 사용되는 binary_crossentropy, metrics는 결과에 반영은 안되고 보여주기만 한다.

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=300, batch_size=4, 
            validation_split=0.2 ,callbacks=[es])

print("======================평가예측======================")
#4. 평가 및 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss[0])
print('accuracy : ', loss[1])
