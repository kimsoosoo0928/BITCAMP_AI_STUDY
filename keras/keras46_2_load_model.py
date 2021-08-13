import numpy as np
# import pandas as pd
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape) #(442,10), (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
""" model = Sequential()
model.add(Dense(500, input_dim=10, activation='relu'))      #활성화 함수, 안써도 디폴트 갑이 있다. 지금은 relu가 성능이 좋다.
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1)) """     # 현재는 마지막 레이어에 activation을 쓰지 않는다

#^ 컴파일과 핏 주석 처리하고 돌렸을때 차이점
#model = load_model('/study/_save/keras46_1_save_model_1.h5')
#! 모델만 저장한것이라 에러가 난다.
model = load_model('/study/_save/keras46_1_save_model_2.h5')
#! 컴파일과 핏을 한 후에 저장한것이라 실행 가능하다.
#! 모델 + 가중치 까지 정장된것이다.(따라서 몇번을 돌려도 같은 값만 나온다.)

#model.summary()


#3. 컴파일, 훈련
#model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
import time

es = EarlyStopping(monitor= 'loss', patience=10, mode='min', verbose=1)

start_time = time.time()

""" model.fit(x_train, y_train, epochs=100, batch_size=18, 
            validation_split=0.2, shuffle=True, verbose=2, callbacks=[es]) """

end_time = time.time() - start_time
print('걸린 시간 : ', end_time)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print('rmse : ' , rmse)

