import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. data
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape) # (4, 3) (4,)

x = x.reshape(4, 3, 1) # (bacth_size, timesteps, feature) 

#2. model
model = Sequential()
model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1))) # units = output, 행무시이기때문에 (3,1)
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
simple_rnn (SimpleRNN)       (None, 10)                120
_________________________________________________________________
dense (Dense)                (None, 10)                110
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
=================================================================
Total params: 241
Trainable params: 241
Non-trainable params: 0
_________________________________________________________________

PARAM = num_units * (num_units + input_dim + 1)
      = 파라미터 아웃값 * (파라미터 아웃값 + 디멘션 값 + 1(바이어스))
      = 10 * (10 + 1 + 1)
      = (input + bias) * output + output * output
      = (input + bias + output) * output
'''

#3. compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1)

#4. evaluate, predict
x_input = np.array([5,6,7]).reshape(1,3,1)
results = model.predict(x_input)
print(results) 


# epochs=1000 : [[8.009891]]
