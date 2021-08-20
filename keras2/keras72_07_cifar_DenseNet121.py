from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
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
densenet121 = DenseNet121(weights='imagenet', include_top=False, input_shape=(32,32,3))   # include_top=False : input_shape 조정 가능

densenet121.trainable=False   # False: vgg훈련을 동결한다(True가 default)

model = Sequential()
model.add(densenet121)
model.add(Flatten())
model.add(Dense(100))        # *layer 1 추가
model.add(Dense(100, activation='softmax'))         # *layer 2 추가

# model.add(densenet121)
# model.add(GlobalAveragePooling2D())
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(100, activation='softmax'))


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
hist = model.fit(x_train, y_train, epochs=3, batch_size=500, validation_split=0.012)
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
time :  143.09228444099426
loss :  0.7564243078231812
acc :  0.755829930305481

cifar10, trainable= T , GAP
time :  62.391611099243164
loss :  0.31332626938819885
acc :  0.8930971622467041

cifar10, trainable= F , FC
time :  22.46883749961853
loss :  1.7269269227981567
acc :  0.481862336397171

cifar10, trainable= F , GAP
time :  22.540725469589233
loss :  1.3336721658706665
acc :  0.5318421125411987

cifar100, trainable= T , FC
time :  61.79895305633545
loss :  1.2539646625518799
acc :  0.6423684358596802

cifar100, trainable= T , GAP
time :  62.17620348930359
loss :  1.2901026010513306
acc :  0.6334818005561829

cifar100, trainable= F , FC
time :  22.615232944488525
loss :  3.5244638919830322
acc :  0.23054656386375427

cifar100, trainable= F , GAP
time :  22.779532194137573
loss :  3.5995888710021973
acc :  0.2266194373369217
'''