weight = 0.5
input = 0.5
goal_prediction = 0.8
lr = 0.01
epochs=100

for iteration in range(epochs): # iteration 은 epochs동안 돈다.
    prediction = input * weight # preddiction은 input * weight 
    error = (prediction - goal_prediction) ** 2 # error = prediction - goal_prediction

    print("Error : " + str(error) + "\tPrediction " + str(prediction)) # error, prediction 출력

    up_prediction = input * (weight + lr) # up_prediction = input *( weight + lr)
    up_error = (goal_prediction - up_prediction) ** 2 # up_error = goal_prediction - up_prediction

    down_prediction = input * (weight -lr)
    down_error = (goal_prediction - down_prediction) ** 2

    if(down_error < up_error) : # weight 감소 
        weight = weight - lr 
    if(down_error > up_error) : # weight 증가
        weight = weight + lr 


