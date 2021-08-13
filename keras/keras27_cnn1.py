from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential() #(N,5,5,1)
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(5, 5, 1))) # (N,4,4,10) # padding의 defalut는 valid이다.
model.add(Conv2D(20, (2,2), activation='relu')) # (N,3,3,20)
model.add(Conv2D(30, (2,2), padding='valid'))
model.add(Flatten()) # (N, 180) 위에서 받아온것이 shape가 2차원이된다.
# 180은 열(특성)이 된다. 

# node를 줄이자!
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 결과 : 3,3,20. * input 생략 가능, kernel_size를 명시안해줘도된다.
# 배치사이즈 단위로 잘라서 훈련시킨다.
# Output = 10
# 이미지를 가로2,세로2로 자른다.
# input_shape=(5, 5, 1) 가로 10 세로 10, 흑백데이터 *행무시
# 5,5,1 : 25개 => 4,4,10 :160개 *
# Dense는 보통 2차원 데이터를 받아들인다.
# (N, 4X4,10) => (N,160) *데이터의 내용물은 바뀌지 않지만, shape만 바꿔준다.
