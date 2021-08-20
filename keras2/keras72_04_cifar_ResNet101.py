from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet101
from tensorflow.python.keras.applications import vgg16     # layer 깊이가 16, 19
from tensorflow.keras.datasets import cifar100

# 동결하고, 안하고 비교
# FC를 모델로 하고, GlobalAveragepooling2D으로 하고

# 1. 데이터
(x_train,y_train), (x_test, y_test) = cifar100.load_data()
# ic(x_train.shape, y_train.shape)   # (50000, 32, 32, 3), (50000, 1)
# ic(x_test.shape, y_test.shape)     # (10000, 32, 32, 3), (10000, 1)

# ic(np.unique(y_train))   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] - 10개
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

# 1-2. 데이터전처리
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
# ic(y_train.shape, y_test.shape)   # (50000, 10), (10000, 10)



# 2. 모델
resnet101 = ResNet101(weights='imagenet', include_top=False, input_shape=(32,32,3))   # include_top=False : input_shape 조정 가능

resnet101.trainable=False   # False: vgg훈련을 동결한다(True가 default)

model = Sequential()
# model.add(resnet101)
# model.add(Flatten())
# model.add(Dense(100))        # *layer 1 추가
# model.add(Dense(100, activation='softmax'))         # *layer 2 추가

model.add(resnet101)
model.add(GlobalAveragePooling2D())
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

# model.trainable=False   # False: 전체 모델 훈련을 동결한다.(True가 default)

model.summary()

print(len(model.weights))               # 26 -> 30(layer 2개 추가 : 2(w+b)=4)
print(len(model.trainable_weights))     # 0 -> 4



# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=3, batch_size=1000, validation_split=0.012)
end_time = time.time() - start


# 4. 평가, 예측



results = model.evaluate(x_test, y_test)

acc = hist.history['accuracy']

loss = hist.history['loss']


print("time : ", end_time)
print('loss : ', loss[2])
print('acc : ', acc[2])



'''
cifar10, trainable= T , FC
time :  67.32410931587219
loss :  0.5866748690605164
acc :  0.7977530360221863

cifar10, trainable= T , GAP

time :  68.34290862083435
loss :  0.5254896879196167
acc :  0.8231174349784851

cifar10, trainable= F , FC
time :  22.05126976966858
loss :  1.2045793533325195
acc :  0.6004048585891724

cifar10, trainable= F , GAP
time :  22.971911430358887
loss :  1.0410091876983643
acc :  0.6330364346504211

cifar100, trainable= T , FC
time :  67.57242727279663
loss :  1.3982219696044922
acc :  0.6096761226654053

cifar100, trainable= T , GAP
time :  69.00305891036987
loss :  1.891108751296997
acc :  0.49955466389656067

cifar100, trainable= F , FC
time :  22.020724296569824
loss :  2.624728202819824
acc :  0.3570040464401245

cifar100, trainable= F , GAP
time :  23.121864795684814
loss :  2.4797215461730957
acc :  0.3700404763221741
'''