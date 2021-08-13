

# binary -> but solve with categorical

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.25
)

# 통상적으로 테스트셋은 증폭하지 않음
test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    '../data/horse-or-human',
    target_size=(150, 150),
    batch_size=800,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

xy_test = train_datagen.flow_from_directory(
    '../data/horse-or-human',
    target_size=(150, 150),
    batch_size=800,
    class_mode='categorical',
    shuffle=True,
    subset='validation'
)

np.save('./_save/_NPY/k59_hh_x_train', arr=xy_train[0][0])
np.save('./_save/_NPY/k59_hh_x_test', arr=xy_test[0][0])
np.save('./_save/_NPY/k59_hh_y_train', arr=xy_train[0][1])
np.save('./_save/_NPY/k59_hh_y_test', arr=xy_test[0][1])

# print(xy_train[0][0]) # 
# print(xy_train[0][1]) # 
print(xy_train[0][0].shape) # (771, 150, 150, 3)
print(xy_train[0][1].shape) # (771, 2)