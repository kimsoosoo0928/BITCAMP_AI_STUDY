import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

'''
- Iris-Setosa : 0
- Iris-Versicolour : 1
- Iris-Virginica : 2
'''

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

#2. 모델 구성
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(4,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax')) # 다중분류위해서 softmax 사용, 원핫인코딩의 결과가 (150,3) 이기때문에 3이어야한다.

#3. 컴파일 및 훈련 + EarlyStopping
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy']) # 이진 분류에 사용되는 binary_crossentropy, metrics는 결과에 반영은 안되고 보여주기만 한다.

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, 
            validation_split=0.2 ,callbacks=[es]) # es 적용

print("======================평가예측======================")
#4. 평가 및 예측
loss = model.evaluate(x_test, y_test) # binary_crossentropy
print('loss : ', loss[0])
print('accuracy : ', loss[1])

print("===============예측==================")
print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)
