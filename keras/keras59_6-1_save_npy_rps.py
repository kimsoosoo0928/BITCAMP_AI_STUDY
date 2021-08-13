import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    '../data/rps',
    target_size=(150, 150),
    batch_size=2500,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

xy_test = train_datagen.flow_from_directory(
    '../data/rps',
    target_size=(150, 150),
    batch_size=2500,
    class_mode='categorical',
    shuffle=True,
    subset='validation'
)

np.save('./_save/_npy/k59_rps_x_train', arr=xy_train[0][0])
np.save('./_save/_npy/k59_rps_y_train', arr=xy_train[0][1])
np.save('./_save/_npy/k59_rps_x_test', arr=xy_test[0][0])
np.save('./_save/_npy/k59_rps_y_test', arr=xy_test[0][1])

print(xy_test[0][0].shape) # (1890, 150, 150, 3)
print(xy_test[0][1].shape) # (1890,)

'''
acc :  0.97052401304245
loss :  0.4285714328289032
val loss :  0.3908296823501587
'''