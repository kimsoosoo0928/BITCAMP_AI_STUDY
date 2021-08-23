# 앞뒤가 똑같은 오토인코더

from enum import auto
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import activations

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),
                    activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

def autoencoder(hidden_layer_size): # deep
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),
                    activation='relu'))
    model.add(Dense(units=256, activation= 'relu'))
    model.add(Dense(units=256, activation= 'relu'))
    model.add(Dense(units=256, activation= 'relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=5) # pca 95%

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

import random

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplot(3, 5, figsize=(20, 7))

# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]),5)

# 원본 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax3, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#  deep 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


plt.tight_layout()
plt.show()