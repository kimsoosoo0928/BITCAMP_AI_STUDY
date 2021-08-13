import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
from tensorflow.python.keras.engine.training import Model

# 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=20, train_size=0.7, shuffle=True)

# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# 모델 구성
model = Sequential()
model.add(Dense(128, input_shape=(10, ), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# 컴파일, 훈련
es = EarlyStopping(monitor='val_loss', mode='auto', patience=10)
cp = ModelCheckpoint(monitor='val_loss', mode='auto', filepath='./_save/ModelCheckPoint/keras48_2_MCP.hdf', save_best_only=True)
model.compile(loss="mse", optimizer="adam", loss_weights=1)

model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[es, cp])

model = load_model('./_save/ModelCheckPoint/keras48_2_MCP.hdf')
# model = load_model('./_save/ModelCheckPoint/keras48_2_model.h5')
# model.save('./_save/ModelCheckPoint/keras48_2_model.h5')

# 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

'''
저장 실행
loss :  2972.394287109375
r2 score :  0.4643822784868399

model
loss :  2979.74755859375
r2 score :  0.4630572712679861

load_model
loss :  2972.394287109375
r2 score :  0.4643822784868399

check point
loss :  2962.762451171875
r2 score :  0.46611797104166386
'''