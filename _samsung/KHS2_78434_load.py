import pandas as pd
from pandas.core.tools.datetimes import Scalar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
import time
import numpy as np
import datetime

# 7/26 삼성전자 시가 예측

# 1. Data
# samsung, sk 필요 데이터 추출 
samsung = pd.read_csv('D:\study\samsung\삼성전자 주가 20210721.csv', sep=',', index_col='일자', header=0, engine='python', encoding='cp949')
# sep : 자료의 구분 기준을 설정 
# index_col : 특정 열 이름으로 인덱스를 지정
# header : 첫행을 열이름으로 쓸 것인지  
samsung = samsung[['시가','고가','저가','종가','거래량']]
samsung = samsung.sort_values(by='일자', ascending=True)
samsung = samsung.query('"2011/01/03" <= 일자 <= "2021/07/21"')
samsung_y = samsung.query('"2011/01/10" <= 일자 <= "2021/07/21"')
print(samsung) # [2601 rows x 5 columns]

sk = pd.read_csv('D:\study\samsung\SK주가 20210721.csv', sep=',', index_col='일자', header=0, engine='python', encoding='cp949')
sk = sk[['시가','고가','저가','종가','거래량']]
sk = sk.sort_values(by='일자', ascending=True)
sk = sk.query('"2011/01/03" <= 일자 <= "2021/07/21"')
print(sk) # [2601 rows x 5 columns]

# pd to np
samsung = samsung.to_numpy()
samsung_y = samsung_y.to_numpy()
sk = sk.to_numpy()

# split function
size = 5

def split_x(dataset, size):
    aaa = []  
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)] 
        aaa.append(subset) 
    return np.array(aaa) 

samsung = split_x(samsung, size) # (2597, 5, 5) dim = 3
samsung_y = split_x(samsung_y, size) # (2592, 5, 5) dim =3
sk = split_x(sk, size) # (2597, 5, 5) dim = 3

samsung = samsung.reshape(2597*5, 5) # dim = 2
samsung_y = split_x(samsung_y, size)
sk = sk.reshape(2597*5, 5) # dim = 2

# x1, x2, y, x1_pred, x2_pred, train_test_split

x1 = samsung[:10000] # (10000, 5)

y = samsung[:10000,0] # 시가
y = y.flatten() # (10000,), [18840. 18600. 18420. ... 47150. 45950. 46900.]

x2 = sk[:10000] # (10000, 5)

x1_pred = samsung[-5:] # (5, 5)
x2_pred = sk[-5:] # (5, 5)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.7, random_state=9)

# scaling
scalar = MinMaxScaler()
scalar.fit_transform(x1_train)
scalar.fit_transform(x2_train)
scalar.transform(x1_test)
scalar.transform(x2_test)
scalar.transform(x1_pred)
scalar.transform(x2_pred)

x1_train = x1_train.reshape(x1_train.shape[0],x1_train.shape[1], 1) 
x2_train = x2_train.reshape(x2_train.shape[0],x2_train.shape[1], 1) 
x1_test = x1_test.reshape(x1_test.shape[0],x1_test.shape[1], 1) 
x2_test = x2_test.reshape(x2_test.shape[0],x2_test.shape[1], 1) 

print(x1_train.shape, x2_train.shape, x1_test.shape, x2_test.shape, y_train.shape, y_test.shape) 
# (7000, 5, 1) (7000, 5, 1) (3000, 5, 1) (3000, 5, 1) (7000,) (3000,)

# 2. modeling

input1 = Input(shape=(5, 1))
xx = LSTM(units=100, activation='relu', return_sequences=True)(input1)
xx = Conv1D(32,2, activation='relu')(xx)
xx = Dense(100, activation='relu')(xx)
output1 = Dense(16, name='output1', activation='relu')(xx)

# 2-2 model2
input2 = Input(shape=(5, 1))
xx = LSTM(units=100, activation='relu', return_sequences=True)(input1)
xx = Conv1D(32,2, activation='relu')(xx)
xx = Dense(100, activation='relu')(xx)
output2 = Dense(16, name='output2', activation='relu')(xx)

merge1 = concatenate([output1, output2]) 
merge2 = Dense(100, activation='relu')(merge1)
merge3 = Dense(16, activation='relu')(merge2)
last_output = Dense(1)(merge3)


model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

# 3. compile
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

model.load_weights('./_save/ModelCheckPoint/stock_weight_save02.h5')

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', restore_best_weights=True)

# mcp = ModelCheckpoint(monitor='val_loss', save_best_only=True, verbose=1, mode='auto', filepath = MCPpath)

start_time = time.time()
# hist = model.fit([x1_train, x2_train], y_train, epochs=50, batch_size=256, verbose=1, callbacks=[es, mcp], validation_split=0.2)
end_time = time.time() - start_time

# model.save_weights('./_save/ModelCheckPoint/stock_weight_save02.h5')

# 4. evaluate
results = model.evaluate([x1_test, x2_test], y_test)
print("걸린시간: ", end_time)
print('loss: ',results[0])
print('acc: ',results[1])

y_pred = model.predict([x1_pred, x2_pred])
# print('y_pred: ', y_pred)
print('5일 뒤 예측 주가: ', y_pred[-1])

'''
걸린시간:  0.0
loss:  73180.953125
acc:  0.0
5일 뒤 예측 주가:  [[78830.28 ]
'''