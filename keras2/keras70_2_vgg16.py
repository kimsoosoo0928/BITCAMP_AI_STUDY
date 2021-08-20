from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16

# model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
vgg16 = model =VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))

model = Sequential()
model.add(vgg16)
model.add(Flatten)
model.add(Dense(10))
model.add(Dense(1))

model.summary()

