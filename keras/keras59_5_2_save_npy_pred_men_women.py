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
    fill_mode='nearest'
)

# 일반적으로 테스트셋은 증폭하지 않음
test_datagen = ImageDataGenerator(rescale=1./255)

x_pred = train_datagen.flow_from_directory(
    '../data/men_women',
    target_size=(150, 150),
    batch_size=100,
    class_mode='binary',
    shuffle=False
)

np.save('./_save/_npy/k59_mw_x_pred', arr=x_pred[0][0])
