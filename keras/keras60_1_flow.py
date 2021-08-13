from tensorflow.keras.datasets import fashion_mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.5,
    fill_mode='nearest'
)

# 1. ImageDataGenerator를 정의
# 2. 파일에서 가져오려면 -> flow_from_directory() // x,y가 튜플 형태로 뭉쳐있다.
# 3. 데이터에서 가져오려면 -> flow() // x,y가 나눠져있다.

augument_size = 100
x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1), # x
    np.zeros(augument_size),  # y
    batch_size=augument_size,
    shuffle=False

).next()                      # iterator 방식으로 반환

print(type(x_data)) # <class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'>
print(type(x_data[0])) # <class 'tuple'>
print(type(x_data[0][0])) # <class 'numpy.ndarray'>
print(x_data[0][0].shape) # (100, 28, 28, 1) -> x값
print(x_data[0][1].shape) # (100,)           -> y값

import matplotlib.pyplot as plt
plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')

plt.show()