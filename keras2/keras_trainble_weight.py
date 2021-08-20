import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. data

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

# 2. model
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

'''
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 3)                 6
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 8
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 3
=================================================================
Total params: 17
Trainable params: 17
Non-trainable params: 0
'''

print(model.weights)
print("=========================================================")
print(model.trainable_weights)
print("=========================================================")

print(len(model.weights))
print(len(model.trainable_weights))