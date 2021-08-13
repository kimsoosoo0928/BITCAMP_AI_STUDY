# 59_3 npy를 이용해서 모델을 완성하시오

# np.save('./_save/_npy/k59_3_train_x.npy', arr=xy_train[0][0])
# np.save('./_save/_npy/k59_3_train_y.npy', arr=xy_train[0][1])
# np.save('./_save/_npy/k59_3_test_x.npy', arr=xy_test[0][0])
# np.save('./_save/_npy/k59_3_test_y.npy', arr=xy_test[0][1])

import numpy as np
x_train = np.load('./_save/_npy/k59_3_train_x.npy')
x_test = np.load('./_save/_npy/k59_3_test_x.npy')
y_train = np.load('./_save/_npy/k59_3_train_x.npy')
y_test = np.load('./_save/_npy/k59_3_train_y.npy')

