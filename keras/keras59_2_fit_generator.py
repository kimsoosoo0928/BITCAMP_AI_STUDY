import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1.데이터
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

xy_train = train_datagen.flow_from_directory(
    '../data/brain/train',
    target_size=(150, 150),
    batch_size=5, 
    class_mode='binary',
    shuffle=True
)

# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../data/brain/test',
    target_size=(150, 150),
    batch_size=5, 
    class_mode='binary'
)

# Found 120 images belonging to 2 classes.


# print(xy_train)
# # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001A226A48550>
# print(xy_train[0][0]) # x값
# print(xy_train[0][1]) # y값
# print(xy_train[0][0].shape, xy_train[0][1].shape)  # (5, 150, 150, 3) (5,)  

# print(xy_train[31][1]) # 마지막 배치 y

# print(type(xy_train))       # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))    # <class 'tuple'>
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')

hist = model.fit_generator(xy_train, epochs=50, steps_per_epoch=32,
                            validation_data=xy_test, # 160/5 =32
                            validation_steps=4)
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

# 위를 시각화 할것

print('acc : ', acc[-1])
print('val_acc : ', acc[-1])