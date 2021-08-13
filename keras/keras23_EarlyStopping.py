import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 한글폰트
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.7,test_size=0.3, shuffle=True, random_state=66) # x 대신 x_scale을 넣어주어야 한다.

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler() # 표준정규분포에 입각한 스탠다는 스케일러 
scaler.fit(x_train) # 반드시 x_train을 fit한다.
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련 
model.compile(loss="mse", optimizer="adam")

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1) # es 정의, 통상적으로 val_loss가 좋다

# loss와 val_loss의 차이가 적으면 과적합이 발생하지 않은 좋은 모델이라고 할 수 있다.

hist = model.fit(x_train, y_train, epochs=100, batch_size=8, 
            validation_split=0.2 ,callbacks=[es]) # es 적용

print(hist.history.keys())
print("============== loss ===============")
print(hist.history['loss'])
print("============== val_loss ==============")
print(hist.history['val_loss'])
print("=============== 평가, 예측 ======================")


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

import matplotlib.pyplot as plt

plt.plot(hist.history['loss']) # x : epoch, y : hist.history['loss']
plt.plot(hist.history['val_loss'])

plt.title("로스, 발로스") # 한글깨짐 해결 과제 !
plt.xlabel('epoch')
plt.ylabel('loss, val_loss')
plt.legend(['train loss', 'val loss']) # 각각의 plot에 매칭되어 범례가 생긴다.
plt.show()


# 5/5 [==============================] - 0s 3ms/step - loss: 10.3596 -> 모델 eva가 출력된것이다.
# 
