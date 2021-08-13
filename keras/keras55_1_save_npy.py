from sklearn import datasets
from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes, load_wine
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100

datasets = load_iris()

x_data = datasets.data
y_data = datasets.target

# print(type(x_data), type(y_data))

np.save('./_save/_npy/k55_x_data_iris.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_iris.npy', arr=y_data)

# 실습 8분동안
# 보스톤, 캔서, 디아벳까지 npy로 세이브

datasets = load_boston()

x_data = datasets.data
y_data = datasets.target

np.save('./_save/_npy/k55_x_data_boston.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_boston.npy', arr=y_data)

datasets = load_breast_cancer()

x_data = datasets.data
y_data = datasets.target

np.save('./_save/_npy/k55_x_data_breast_cancer.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_breast_cancer.npy', arr=y_data)

datasets = load_diabetes()

x_data = datasets.data
y_data = datasets.target
np.save('./_save/_npy/k55_x_data_diabet.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_diabet.npy', arr=y_data)


datasets = load_wine()

x_data = datasets.data
y_data = datasets.target


np.save('./_save/_npy/k55_x_data_wine.npy', arr=x_data)
np.save('./_save/_npy/k55_y_data_wine.npy', arr=y_data)

########################################################################



(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(60000, 28 * 28)
np.save('./_save/_npy/k55_x_data_train_mnist.npy', arr=x_train)
np.save('./_save/_npy/k55_y_data_train_mnist.npy', arr=y_train)
np.save('./_save/_npy/k55_x_data_test_mnist.npy', arr=x_test)
np.save('./_save/_npy/k55_y_data_test_mnist.npy', arr=y_test)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# x_train = x_train.reshape(60000, 28 * 28)
np.save('./_save/_npy/k55_x_data_train_fashion.npy', arr=x_train)
np.save('./_save/_npy/k55_y_data_train_fashion.npy', arr=y_train)
np.save('./_save/_npy/k55_x_data_test_fashion.npy', arr=x_test)
np.save('./_save/_npy/k55_y_data_test_fashion.npy', arr=y_test)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

np.save('./_save/_npy/k55_x_data_train_cifar10.npy', arr=x_train)
np.save('./_save/_npy/k55_y_data_train_cifar10.npy', arr=y_train)
np.save('./_save/_npy/k55_x_data_test_cifar10.npy', arr=x_test)
np.save('./_save/_npy/k55_y_data_test_cifar10.npy', arr=y_test)

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

np.save('./_save/_npy/k55_x_data_train_cifar100.npy', arr=x_train)
np.save('./_save/_npy/k55_y_data_train_cifar100.npy', arr=y_train)
np.save('./_save/_npy/k55_x_data_test_cifar100.npy', arr=x_test)
np.save('./_save/_npy/k55_y_data_test_cifar100.npy', arr=y_test)