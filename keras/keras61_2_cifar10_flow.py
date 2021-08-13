# 훈련데이터를 10만개를 증폭할것!!
# 완료후 기존 모델과 비교
# save_dir도 temp에 넣을것
# 증폭데이터는 temp에 저장후 룬련 끝난후 결과 본뒤 삭제

from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, 
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.5,
    fill_mode='nearest'
)

# 1. ImageDataGenerator를 정의
# 2. 파일에서 가져오려면 -> flow_from_directory() // x,y가 튜플 형태로 뭉쳐있다.
# 3. 데이터에서 가져오려면 -> flow() // x,y가 나눠져있다.

augment_size = 50000

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(x_train.shape[0]) # 60000
print(randidx)          # [20693 47880 21722 ... 50370 50531 26723]
print(randidx.shape)    # (40000,)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()
print(x_augmented.shape) # (40000, 28, 28)

x_augmented = x_augmented.reshape(x_augmented.shape[0], 32, 32, 3)
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

x_augmented = train_datagen.flow(x_augmented, np.zeros(augment_size),
                                 batch_size=augment_size, shuffle=False).next()[0]

print(x_augmented.shape) # (40000, 28, 28)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)

# 모델 완성 !!!
# 비교대상? loss, val_loss, acc, val_acc
# 기존 fashion_mnist와 결과비교 

# from model

x_train = x_train.reshape(100000, 32*32*3) # (100000, 28, 28, 1)
x_test = x_test.reshape(10000, 32*32*3) # (10000, 28, 28, 1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(100000, 32*32, 3) # (100000, 28, 28, 1)
x_test = x_test.reshape(10000, 32*32, 3) # (10000, 28, 28, 1)

# one hot encoding
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (100000, 10)
y_test = one.transform(y_test).toarray() # (10000, 10)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, GlobalAveragePooling1D, Dropout

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, padding='same',                        
                        activation='relu' ,input_shape=(32*32, 3))) 
model.add(Conv1D(32, 2, padding='same', activation='relu'))                   
model.add(MaxPool1D())                                         
model.add(Conv1D(64, 2, padding='same', activation='relu'))                   
model.add(Conv1D(64, 2, padding='same', activation='relu'))    
model.add(MaxPool1D())                                         
model.add(Conv1D(128, 3, padding='same', activation='relu'))                   
model.add(Conv1D(128, 3, padding='same', activation='relu'))    
model.add(Flatten())                                              
model.add(Dense(256, activation='relu'))
model.add(Dense(124, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. complie
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

import time 

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=576, verbose=2,
    validation_split=0.05, callbacks=[es])
end_time = time.time() - start_time

# 4. eval

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print('acc : ', acc[-10])
print('val_acc : ', val_acc[-10])
print('val_loss : ', val_loss[-10])

