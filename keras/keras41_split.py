import numpy as np

a = np.array(range(1,11))
size = 5

# 시계열 데이터의 X와 Y 분리시키는 함수 분석하기

def split_x(dataset, size):
    aaa = [] # list 설정 
    for i in range(len(dataset) - size + 1): # 데이터셋의 길이 - 5 + 1 동안
        subset = dataset[i : (i + size)] # 서브셋은 데이터셋의 i부터 i + size
        aaa.append(subset) # 리스트 aaa에 서브셋 추가
    return np.array(aaa) # 넘파이 어레이로 리스트 반환 

dataset = split_x(a, size)

print(dataset)

x = dataset[:, :4]
y = dataset[:, 4]

print("x : \n", x)
print("y : ", y)

'''
x :
 [[1 2 3 4]
 [2 3 4 5]
 [3 4 5 6]
 [4 5 6 7]
 [5 6 7 8]
 [6 7 8 9]]
y :  [ 5  6  7  8  9 10]
'''