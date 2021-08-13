# 과제 01
#1. R2를 음수가 아닌 0.5 이하로 만들어라.
#2. 데이터 건들지 말것
#3. 레이어는 인풋 아웃풋 포함 6개 이상
#4. batch_size = 1
#5. epochs는 100 이상
#6. 히든레이어의 노드는 10개 이상 1000개 이하 
#7. train 70% 

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import random


#1. 데이터
x = np.array(range(100)) #(100,)
y = np.array(range(1,101)) #(100,)
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, shuffle=True, random_state=66) # 70% 데이터를 train으로 준다. random_state를 설정해줌으로써 값을 고정시켜준다.
print(x_test) # 30개
print(y_test) # 30개 

random.setstate

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(2))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='KLDivergence', optimizer='RMSprop')
model.fit(x_train, y_train, epochs=100, batch_size=1) # fit에서는 train data를 사용해야한다.

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) # evaluate에서는 평가용 데이터를 사용한다. 
print('loss : ', loss)


y_predict = model.predict(x_test)
print('x의 예측값 : ', y_predict)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("r2스코어 : ", r2)

'''
loss :  0.029051128774881363
x의 예측값 :  [[ 5.1981325 ]
 [55.98377   ]
 [ 2.8082216 ]
 [ 3.4057    ]
 [31.487167  ]
 [24.914911  ]
 [ 0.41830942]
 [44.03421   ]
 [52.99638   ]
 [41.046818  ]
 [15.355259  ]
 [11.172914  ]
 [15.952738  ]
 [17.745173  ]
 [39.85186   ]
 [30.292213  ]
 [48.21655   ]
 [27.304821  ]
 [23.12247   ]
 [35.072033  ]
 [29.694735  ]
 [51.203945  ]
 [56.581245  ]
 [52.3989    ]
 [ 9.380481  ]
 [ 2.2107437 ]
 [ 8.783002  ]
 [20.135082  ]
 [14.160305  ]
 [14.757784  ]]
r2스코어 :  0.46973882299772873
'''