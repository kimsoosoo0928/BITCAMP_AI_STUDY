from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import Xception
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
xception = Xception(weights='imagenet', include_top=False, input_shape=(32, 32, 3))   # include_top=False : input_shape 조정 가능

xception.trainable=True   # False: vgg훈련을 동결한다(True가 default)

model = Sequential()
model.add(xception)
model.add(Flatten())
model.add(Dense(100))        # *layer 1 추가
model.add(Dense(10, activation='softmax'))         # *layer 2 추가

# model.trainable=False   # False: 전체 모델 훈련을 동결한다.(True가 default)

# model.add(xception)
# model.add(GlobalAveragePooling2D())
# model.add(Dense(2048, activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(100, activation='softmax'))

model.trainable=True   # False: 전체 모델 훈련을 동결한다.(True가 default)


model.summary()

print(len(model.weights))               # 26 -> 30(layer 2개 추가 : 2(w+b)=4)
print(len(model.trainable_weights))     # 0 -> 4



# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)

import time
start = time.time()
model.fit(x_train, y_train, epochs=3, batch_size=3000, validation_split=0.012)
end = time.time() - start


# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('걸린시간 :', end)
print('category :', results[0])
print('accuracy :', results[1])


# cifar 10, trainable = True, FC 
# 걸린시간 : 147.90595054626465
# category : 1.7997206449508667
# accuracy : 0.24619999527931213

# cifar 10, trainable = False, FC
# 걸린시간 : 51.481168270111084
# category : 1.6238356828689575
# accuracy : 0.5350000262260437

# cifar 10, trainable = True, GAP
# 걸린시간 : 147.25001335144043
# category : 2.3025901317596436
# accuracy : 0.10000000149011612

# cifar 10, trainable = False, GAP
# 걸린시간 : 55.336286783218384
# category : 1.1168488264083862
# accuracy : 0.6205999851226807

# cifar 100, trainable = True, FC 
# 걸린시간 : 77.14194703102112
# category : 4.605197906494141
# accuracy : 0.009999999776482582

# cifar 100, trainable = True, GAP
# 걸린시간 : 24.472861528396606
# category : 11.365644454956055
# accuracy : 0.007499999832361937

# cifar 100, trainable = False, FC
# 걸린시간 : 23.954719066619873
# category : 52.542579650878906
# accuracy : 0.011500000022351742

# cifar 100, trainable = False, GAP
# 걸린시간 : 24.55642008781433
# category : 13.382279396057129
# accuracy : 0.014100000262260437