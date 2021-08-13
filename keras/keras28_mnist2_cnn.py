import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)/255. # (60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)/255. # (10000, 28, 28, 1)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (60000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.python.keras.layers.core import Flatten

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5),                          
                        padding='same', activation='relu' ,input_shape=(28, 28, 1))) 
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))                   
model.add(MaxPool2D())                                         
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))                   
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))    
model.add(Flatten())                                              
model.add(Dense(256, activation='relu'))
model.add(Dense(124, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. comple fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=10000, batch_size=576, verbose=2,
    validation_split=0.0005, callbacks=[es])

# 4. predict eval -> 

loss = model.evaluate(x_test, y_test)
print('loss[category] : ', loss[0])
print('loss[accuracy] : ', loss[2])
