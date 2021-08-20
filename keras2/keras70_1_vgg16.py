from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16

# model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
model =VGG16(weights='imagenet', include_top=False, 
# input_shape=(100,100,3)
)


model.trainable=False 

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))


# FC 용어 정리
# Fully Connected layer : 한층의 모든 뉴런이 그 다음 층의 모든 뉴런과 연결된 상태 