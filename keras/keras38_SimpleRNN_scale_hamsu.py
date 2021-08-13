import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Input

#1. data
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape, y.shape) # (13, 3) (13,)

x = x.reshape(13, 3, 1) # (bacth_size, timesteps, feature) 
x = x.reshape(x.shape[0], x.shape[1], 1)


#2. model
'''model = Sequential()
model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
# model.add(GRU(units=32, activation='relu', input_shape=(3,1))) # units = output, 행무시이기때문에 (3,1)
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))'''

input = Input(shape=(3,1))
dense = LSTM(units=32, activation='relu')(input)
dense1 = Dense(16)(dense)
dense2 = Dense(8)(dense1)
dense3 = Dense(4)(dense2)
dense4 = Dense(2)(dense3)
output = Dense(1)(dense4)

model = Model(inputs=input, outputs=output)

model.summary()

#3. compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. evaluate, predict
x_input = np.array([5,6,7]).reshape(1,3,1)
y_pred = model.predict(x_input)
print(y_pred)

'''
함수형 : [[7.8914647]]
'''