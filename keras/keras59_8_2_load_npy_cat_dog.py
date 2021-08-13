import numpy as np
import time

# load data
x_train = np.load('./_save/_npy/k59_cd_x_train.npy')
y_train = np.load('./_save/_npy/k59_cd_y_train.npy')
x_test = np.load('./_save/_npy/k59_cd_x_test.npy')
y_test = np.load('./_save/_npy/k59_cd_y_test.npy')


# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(3,3), input_shape =(150,150,3), activation= 'relu'))
model.add(Conv2D(filters = 32, kernel_size=(3,3),  activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters = 128, kernel_size=(3,3),  activation= 'relu'))
model.add(Conv2D(filters = 128, kernel_size=(3,3),  activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(2, activation= 'sigmoid'))

# 3. compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience=7, mode='auto', verbose=1)

hist = model.fit(x_train, y_train, epochs=50,
                validation_split=0.2,
                steps_per_epoch=32,
                validation_steps=4,
                callbacks=[es])

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print('acc : ', acc[-1])
print('loss : ', loss[0])
print('val loss : ', val_acc[-1])

'''
acc :  0.8330000042915344
loss :  1.0598777532577515
val loss :  0.550000011920929
'''