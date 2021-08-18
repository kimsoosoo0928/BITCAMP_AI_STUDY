import numpy as np 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
from tensorflow.python.keras import activations
from tensorflow.python.keras.backend import dropout
from tensorflow.python.keras.layers.recurrent_v2 import LSTM
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD 

#1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255
# print(x_train.shape, x_test.shape) #(60000, 28, 28, 1) (10000, 28, 28, 1)

# 2. 모델 
def build_model(drop='dropout', optimizer='adam',node='node',activation='activation',lr='lr',model='model'):
    inputs = Input(shape=(28*28), name='input')
    x = model(node, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(node, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(node, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='ouput')(x)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = optimizer(lr)
    model.compile(optimizer=optimizer, metrics=['acc'],
                    loss='categorical_crossentropy')
    return model 

# x_train = x_train.reshape(60000, 28,28,1).astype('float32')/255
# x_test = x_test.reshape(10000, 28,28,1).astype('float32')/255
# def build_model(drop='dropout', optimizer='adam',node='node',activation='activation',lr='lr'):
#     inputs = Input(shape=(28,28,1), name='input')
#     x = Conv2D(node,kernel_size=(2,2), padding='valid', name='hidden1')(inputs)
#     x = Flatten()(x)
#     x = Dense(node, activation=activation, name='hidden2')(x)
#     x = Dropout(drop)(x)
#     x = Dense(node, activation=activation, name='hidden3')(x)
#     x = Dropout(drop)(x)
#     outputs = Dense(10, activation='softmax', name='ouput')(x)
#     model = Model(inputs=inputs, outputs=outputs)
#     optimizer = optimizer(lr)
#     model.compile(optimizer=optimizer, metrics=['acc'],
#                     loss='categorical_crossentropy')
#     return model 

def create_hyperparameter():
    batches = [1000, 2000]
    optimizers = [Adam, SGD,]
    lr = [0.01,0.002,0.05]
    activation = ['relu']
    dropout = [0.1, 0.2, 0.3]
    node = [10,20]
    epochs = [1,2,3]
    model =[Dense, LSTM]
    return {"batch_size" : batches, "optimizer" : optimizers, "lr" : lr, "drop": dropout, 
    "node": node, "activation":activation, "epochs":epochs, "model":model}

hyperparameters = create_hyperparameter()
print(hyperparameters)
# {'batch_size': [10, 20, 30, 40, 50], 'optimizer': ['rmsprop', 'adam', 'adadelta'], 'drop': [0.1, 0.2, 0.3]}

# model2 = build_model() 

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier # 텐서플로우 모델을 사이킥런 모델로 인식하게 만들어주는것 
model2 = KerasClassifier(build_fn=build_model, verbose=1, validation_split=0.2)
#keras모델을 kerasclassfier에 랩핑해주면 사이킥런에 사용할 수 있다 
#KerasClassifier = epochs 사용 가능 validation_split 사용가능 

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
model = RandomizedSearchCV(model2, hyperparameters, cv=5)
#RandomizedSearchCV 기존에는 사이칵런 머신러닝만 넣어줫고 텐서플로우모델을 넣은적이 없다
#텐서플로우를 사이킥런에서 사용하고싶으면 어케해야할까? 
#텐서플로우 형식을 사이킥런 형식으로 바꿔주면 된다 
#텐서플로우 모델을 사이킥런으로 감싸주면 된다 -> 랩핑 하면된다 
#cv=5 kfold값을 그냥 넣어주면 알아서 kfold와 동일한 효과를 준다 

model.fit(x_train, y_train, verbose=1)#, validation_split=0.2) # ,epochs=3 

#KerasClassifier 보다 fit epochs가 우선 순위 
#1280/1280 [==============================] - 4s 3ms/step - loss: 2.2220 - acc: 0.2137 - val_loss: 2.1122 - val_acc: 0.4776

print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
acc = model.score(x_test, y_test)
print("최종 스코어 : ", acc)
