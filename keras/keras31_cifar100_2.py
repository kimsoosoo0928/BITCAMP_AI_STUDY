import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100


# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data() # (50000, 32, 32, 3) (50000, 1), (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32, 32, 3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3) # (10000, 28, 32, 3)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() 
y_test = one.transform(y_test).toarray() 

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.python.keras.layers.core import Flatten

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(2, 2),                          
                        padding='same', activation='relu', 
                        input_shape=(32, 32, 3))) 
model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))                   
model.add(MaxPool2D())                                         
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))                   
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))    
model.add(MaxPool2D())                                         
model.add(Conv2D(128, (4, 4), activation='relu'))                   
model.add(Conv2D(128, (4, 4), activation='relu'))
model.add(Flatten())                                              
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. comple fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=25, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=10000, batch_size=512, verbose=2,
    validation_split=0.0005, callbacks=[es])

# 4. predict eval 

loss = model.evaluate(x_test, y_test)
print('loss[category] : ', loss[0])
print('loss[accuracy] : ', loss[2])

'''
epochs=10000, batch_size=512, patience=25
loss[accuracy] :  0.4325000047683716
'''