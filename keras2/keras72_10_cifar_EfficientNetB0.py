from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
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
b0 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32,32,3))   # include_top=False : input_shape 조정 가능

b0.trainable=False  # False: vgg훈련을 동결한다(True가 default)

model = Sequential()
# model.add(b0)
# model.add(Flatten())
# model.add(Dense(100))        # *layer 1 추가
# model.add(Dense(100, activation='softmax'))         # *layer 2 추가


model.add(b0)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

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
time :  30.991266012191772
loss :  0.5754619240760803
acc :  0.796761155128479

cifar10, trainable= T , GAP
time :  42.114800453186035
loss :  0.39977630972862244
acc :  0.8600000143051147

cifar10, trainable= F , FC
time :  18.502704858779907
loss :  1.2897320985794067
acc :  0.5461133718490601

cifar10, trainable= F , GAP
time :  18.524895191192627
loss :  1.26948881149292
acc :  0.5475910902023315

cifar100, trainable= T , FC
time :  42.140533208847046
loss :  1.3322303295135498
acc :  0.621052622795105

cifar100, trainable= T , GAP
time :  42.525267601013184
loss :  1.4387757778167725
acc :  0.5941497683525085

cifar100, trainable= F , FC
time :  18.566548109054565
loss :  3.116495132446289
acc :  0.24720647931098938

cifar100, trainable= F , GAP

time :  18.683337688446045
loss :  3.0969369411468506
acc :  0.2389068752527237
'''