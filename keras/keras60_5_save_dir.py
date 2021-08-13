# save_dir 설명
# flow 또는 flow_directory 의 iterater 구조

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

augment_size = 10

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(x_train.shape[0]) # 60000
print(randidx)          # [20693 47880 21722 ... 50370 50531 26723]
print(randidx.shape)    # (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape) # (40000, 28, 28)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

import time
start_time = time.time()
x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                 batch_size=augment_size, shuffle=False,
                                 save_to_dir='d:/temp/' # 이번파일은 save_to_dir이 main!
                                 )#.next()[0]
end_time = time.time()-start_time
print(x_augmented.shape) # (40000, 28, 28)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)

print("걸린시간 : ", end_time)

# 실습 1. x_augmented 10개와 원래 x_train 10개를 비교하는 이미지를 출력할것
#         subplot(2, 10, ?) 사용
#         2시까지
