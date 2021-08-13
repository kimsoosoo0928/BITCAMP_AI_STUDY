import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1. data
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
            [5,6,7],[6,7,8],[7,8,9],[8,9,10],
            [9,10,11],[10,11,12],
            [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = np.array([50,60,70]).reshape(1,3,1)
# -> (None, 3, 1)

print(x.shape, y.shape) # (13, 3) (13,)

x = x.reshape(13, 3, 1) # (bacth_size, timesteps, feature) 
x = x.reshape(x.shape[0], x.shape[1], 1)


#2. model
model = Sequential()
model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
# model.add(GRU(units=32, activation='relu', input_shape=(3,1))) # units = output, 행무시이기때문에 (3,1)
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.summary()

#3. compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. evaluate, predict
results = model.predict(x_predict)
print(results) 

'''
[[80.5121]]
model = Sequential()
model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
# model.add(GRU(units=32, activation='relu', input_shape=(3,1))) # units = output, 행무시이기때문에 (3,1)
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

model.summary()

#3. compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)
'''