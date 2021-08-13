import time
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, SimpleRNN, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error


#1. Data Preprocessing
x_data = np.array(range(1, 101))
x_pred = np.array(range(96, 106))

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)
dataset = split_x(x_data, 6)

x = dataset[:, :5] # (95, 5) 
y = dataset[:, 5] # (95,)
# print(x.shape, y.shape)
x_pred = split_x(x_pred, 5) # (6, 5)
# print(x_pred, x_pred.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=66) 
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1], 1)

#2. Modeling
input = Input(shape=(5, 1))
s = LSTM(16, activation='relu')(input)
s = Dense(64, activation='relu')(s)
s = Dense(32, activation='relu')(s)
s = Dense(8, activation='relu')(s)
s = Dense(4, activation='relu')(s)
output = Dense(1, activation='relu')(s)

model = Model(inputs=input, outputs=output)

#3. Compiling, Training
es = EarlyStopping(monitor='val_loss', patience=64, mode='min', verbose=1)
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=128, batch_size=16, validation_split=0.05, callbacks=[es])
end_time = time.time() - start_time

#. Evaluating, Prediction
mse = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
result = model.predict(x_pred)

print('mse : ', mse)
print('rmse : ', rmse)
print('R2 score = ', r2)
print('pred : ', result)
print('time taken(s) : ', end_time)