from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,) (10000,)

print("x[0] 값 : ", x_train[3])
print("y[0] 값 : ", y_train[3])

plt.imshow(x_train[3], 'gray')
plt.show()
