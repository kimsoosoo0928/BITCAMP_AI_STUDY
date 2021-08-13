from tensorflow.keras.datasets import cifar100

# 완성하시오!!

import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) (10000, 1)

print("x[0] 값 : ", x_train[3])
print("y[0] 값 : ", y_train[3])

plt.imshow(x_train[3], 'gray')
plt.show()