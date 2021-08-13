import numpy as np

# load data
x_train = np.load('./_save/_npy/k59_mw_x_train.npy')
y_train = np.load('./_save/_npy/k59_mw_y_train.npy')
x_test = np.load('./_save/_npy/k59_mw_x_test.npy')
y_test = np.load('./_save/_npy/k59_mw_y_test.npy')
x_pred = np.load('./_save/_npy/k59_mw_x_pred.npy')

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
model.add(Dense(1, activation= 'sigmoid'))

# 3. compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit(x_train, y_train, epochs=50,
                validation_split=0.2,
                steps_per_epoch=32,
                validation_steps=4)

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print('acc : ', acc[-1])
print('loss : ', loss[0])

y_predict = model.predict([x_pred])
res = (1-y_predict) * 100
print('남자일 확률 : ', res, '%')
