from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2 
)
# num_words : 단어사전의 개수

print(x_train[0], type(x_train[0])) # <class 'list'>
# print(x_train[1], type(x_train[1])) 
print(y_train[0]) # 3 * 문장의 결과가 3번이다.

print(len(x_train[0]), len(x_train[1])) # 87, 56 *앞에서부터 패딩 필요

# print(x_train[0].shape) # 판다스, 넘파이는 shape가 찍히지만, list는 shape가 찍히지 않는다.
# AttributeError: 'list' object has no attribute 'shape'

print(x_train.shape, x_test.shape) # (8982,) (2246,)
print(y_train.shape, y_test.shape) # (8982,) (2246,)

print(type(x_train)) # <class 'numpy.ndarray'>

# 모델링이 돌아가기 위해선 길이를 전부 맞춰줘야 한다.

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)) # 뉴스기사의 최대길이 : 2376
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train)) # 뉴스기사의 평균길이 :  145.5

# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=500, padding='pre') # (8982, 100) #maxlen은 임의로 정해준다.
x_test = pad_sequences(x_test, maxlen=500, padding='pre') # (8982, 100) 
print(x_train.shape, x_test.shape) 
print(type(x_train), type(x_train[0]))
print(x_train[1])

# y 확인
print(np.unique(y_train))

'''
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45] - 46개의 label
'''

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape, y_test.shape) # (8982, 46) (2246, 46)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, GRU

# 실습, 완성해보세요
# 임베딩으로 시작, 파라미터 2~3개 사용
# 2번째 레이어는 LSTM
# 마지막 레이어는 Dense output = 46개
# softmax, categorical ent, 

model = Sequential()
model.add(Embedding(10000, 100))
model.add(GRU(128))
model.add(Dense(2, activation='sigmoid'))

model.summary()
#3. compile, fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=512, validation_split=0.2, verbose=1)
    
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : ", loss[0])
print("acc : ", loss[1])

'''
loss :  0.5763950347900391
acc :  0.6892399787902832
'''