# 앞뒤가 똑같은 오토인코더

from enum import auto
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float')/255
x_test = x_test.reshape(10000, 784).astype('float')/255

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input_imgs = Input(shape=(784,))

encoded = Dense(1064, activation='relu')(input_imgs)

decoded = Dense(784, activation='sigmoid')(encoded)

# 중요하지 않은 특성들은 사라진다. (주근깨, 점, 등등...)

autoencoder = Model(input_imgs, decoded)

# autoencoder.summary()

# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2)

decoded_imgs = autoencoder.predict(x_test)


import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()