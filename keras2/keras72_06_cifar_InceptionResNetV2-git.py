from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.python.keras.applications import vgg16     # layer 깊이가 16, 19
from tensorflow.keras.datasets import cifar10

# 동결하고, 안하고 비교
# FC를 모델로 하고, GlobalAveragepooling2D으로 하고

# 1. 데이터
(x_train,y_train), (x_test, y_test) = cifar10.load_data()
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
inceptionvresnetv2 = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(32,32,3))   # include_top=False : input_shape 조정 가능

inceptionvresnetv2.trainable=True   # False: vgg훈련을 동결한다(True가 default)

model = Sequential()
model.add(inceptionvresnetv2)
model.add(Flatten())
model.add(Dense(100))        # *layer 1 추가
model.add(Dense(10, activation='softmax'))         # *layer 2 추가

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
hist = model.fit(x_train, y_train, epochs=3, batch_size=3000, validation_split=0.012)
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


cifar10, trainable= T , GAP


cifar10, trainable= F , FC

cifar10, trainable= F , GAP


cifar100, trainable= T , FC


cifar100, trainable= T , GAP


cifar100, trainable= F , FC


cifar100, trainable= F , GAP

'''