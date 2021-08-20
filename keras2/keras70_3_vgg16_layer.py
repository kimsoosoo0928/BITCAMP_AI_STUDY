from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16

# model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
vgg16 = model =VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))

vgg16.trainable=False # vgg훈련을 동결한다.

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))

# model.trainable=False # 훈련을 동결한다.

model.summary()

print(len(model.weights)) 
print(len(model.trainable_weights))
# 26 -> 30
# 0 -> 4

###################### 2번 파일에서 아래만 추가 ####################

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns= ['Layer Type', 'Layer Name', 'Layer Trainable'])

print(results)