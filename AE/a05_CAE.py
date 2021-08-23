# 2번 카피해서 복붙
# cnn으로 딥하게 구성 
# 2개의 모델을 구성하는데 하나는 기본적 오토인코더
# 다른 하나는 딥하게 만든 구성
# 2개 성능 비교 

# Conv2D
# MaxPool
# Conv2D
# MaxPool
# Conv2D -> encoder

# Conv2D
# UpSampling2D
# Conv2D
# UpSampling2D
# Conv2D
# UpSampling2D
# Conv2D(1, ) -> decoder