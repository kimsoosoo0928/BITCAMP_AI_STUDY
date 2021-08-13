import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# 트레인 데이터 증폭
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

# 일반적으로 테스트 데이터는 증폭하지 않음


xy_train = train_datagen.flow_from_directory(
    '../data/cat_dog/training_set/training_set',
    target_size=(150, 150),
    batch_size=2500,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

# Found 6004 images belonging to 2 classes.

xy_test = train_datagen.flow_from_directory(
    '../data/cat_dog/test_set/test_set',
    target_size=(150, 150),
    batch_size=2500,
    class_mode='categorical',
    shuffle=True,
    subset='validation'
)


# Found 505 images belonging to 2 classes.

np.save('./_save/_npy/k59_cd_x_train', arr=xy_train[0][0])
np.save('./_save/_npy/k59_cd_y_train', arr=xy_train[0][1])
np.save('./_save/_npy/k59_cd_x_test', arr=xy_test[0][0])
np.save('./_save/_npy/k59_cd_y_test', arr=xy_test[0][1])

print(xy_train[0][0].shape) # (2500, 150, 150, 3)
print(xy_train[0][1].shape) # (2500, 2)

