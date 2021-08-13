# 훈련데이터를 기존데이터 20% 더 할것
# 성과비교
# save_dir도 temp에 넣을것
# 증폭데이터는 temp에 저장후 룬련 끝난후 결과 본뒤 삭제

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# 1. data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

x_train = np.load('./_save/_npy/k59_mw_x_train.npy')
y_train = np.load('./_save/_npy/k59_mw_y_train.npy')
x_test = np.load('./_save/_npy/k59_mw_x_test.npy')
y_test = np.load('./_save/_npy/k59_mw_y_test.npy')

augment_size = 400 

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(x_train.shape[0]) # 60000
print(randidx)          # [20693 47880 21722 ... 50370 50531 26723]
print(randidx.shape)    # (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape) # (40000, 28, 28)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 150, 150, 3)
x_train = x_train.reshape(x_train.shape[0], 150, 150, 3)
x_test = x_test.reshape(x_test.shape[0], 150, 150, 3)

x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                 batch_size=augment_size, shuffle=False).next()[0]

print(x_augmented.shape) # (40000, 28, 28)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

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
print('loss : ', loss[-1])
print('val loss : ', val_acc[-1])

'''
acc :  0.42603036761283875
loss :  0.42563483119010925
val loss :  0.41941073536872864
'''