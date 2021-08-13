import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names) # (569, 30)

x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(569, 30) (569,) input=30 output=1
print(y[:20])
print(np.unique) # y라는 데이터는 0,1로만 이루어져있다datetime A combination of a date and a time. Attributes: ()

# 실습 : 모델 시작 !

#1. 데이터
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7,test_size=0.3, shuffle=True, random_state=8)

#1-1. 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(30,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 값을 0과 1사이로 한정지어주는 sigmod 함수, 

#3. 컴파일 및 훈련 + EarlyStopping
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy']) # 이진 분류에 사용되는 binary_crossentropy, metrics는 결과에 반영은 안되고 보여주기만 한다.

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=5, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=100, batch_size=8, 
            validation_split=0.2 ,callbacks=[es]) # es 적용

print("======================평가예측======================")
#4. 평가 및 예측
loss = model.evaluate(x_test, y_test) # binary_crossentropy
print('loss : ', loss[0])
print('accuracy : ', loss[1])

print("===============예측==================")
print(y_test[-5:-1])
y_predict = model.predict(x_test[-5:-1])
print(y_predict)


# y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print('r2 score : ', r2)

# import matplotlib.pyplot as plt

# plt.plot(hist.history['loss']) 
# plt.plot(hist.history['val_loss'])

# plt.title("loss, val_loss") 
# plt.xlabel('epoch')
# plt.ylabel('loss, val_loss')
# plt.legend(['train loss', 'val loss']) 
# plt.show()