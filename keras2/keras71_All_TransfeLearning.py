# pre-trained model

from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7

# model = Xception()
# model = VGG16()
# model = VGG19()
# model = ResNet101()
# model = ResNet101V2()
model = ResNet152()
# model = ResNet152V2()
# model = ResNet50()
# model = ResNet50V2()
# model = InceptionV3()
# model = InceptionResNetV2()
# model = MobileNet()
# model = MobileNetV2()
# model = MobileNetV3Large()
# model = MobileNetV3Small()
# model = NASNetLarge()
# model = NASNetMobile()
# model = EfficientNetB0()
# model = EfficientNetB1()
# model = EfficientNetB7()

model.trainable = False

model.summary()

print(' 전체 가중치 갯수 : ', len(model.weights))
print('훈련가능 가중치 갯수 : ', len(model.trainable_weights))

# 모델별로 파라미터와 웨이트 수들 정리

'''
model = Xception()
Total params: 22,910,480
Trainable params: 0
Non-trainable params: 22,910,480
전체 가중치 갯수 :  236
훈련가능 가중치 갯수 :  0

model = VGG16()
Total params: 138,357,544
Trainable params: 0
Non-trainable params: 138,357,544
전체 가중치 갯수 :  32
훈련가능 가중치 갯수 :  0

model = VGG19()
Total params: 143,667,240
Trainable params: 0
Non-trainable params: 143,667,240
전체 가중치 갯수 :  38
훈련가능 가중치 갯수 :  0

model = ResNet101()
Total params: 44,707,176
Trainable params: 0
Non-trainable params: 44,707,176
전체 가중치 갯수 :  626
훈련가능 가중치 갯수 :  0

model = ResNet101V2()
Total params: 44,675,560
Trainable params: 0
Non-trainable params: 44,675,560
전체 가중치 갯수 :  544
훈련가능 가중치 갯수 :  0
'''

