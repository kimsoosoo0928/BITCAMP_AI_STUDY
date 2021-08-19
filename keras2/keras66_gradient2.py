import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6

gradient = lambda x: 2*x -4 

x0 = 10.0 # 초기값
MaxIter = 20
learning_rate = 0.25

print("step\tx\tf(x)")
print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))

for i in range(MaxIter):
    x1 = x0 - learning_rate * gradient(x0)
    x0 = x1

    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0)))


# 초기값 변경
# 러닝레이트 변경 