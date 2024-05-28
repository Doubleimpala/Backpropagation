import math
import random
import numpy as np

bias = [0, 0]
weight = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# learning rate
epsilon = 0.7
# momentum
alpha = 0.3
error = 1

# training data
train1inp1 = 0
train1inp2 = 0
train1out1 = 0

train2inp1 = 1
train2inp2 = 0
train2out1 = 1

train3inp1 = 0
train3inp2 = 1
train3out1 = 1

train4inp1 = 1
train4inp2 = 1
train4out1 = 0
# end training data

input1 = 0
input2 = 0
output1 = 0


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoidprime(x):
    return math.exp(-x) / (1 + math.exp(-x)) ** 2


def evaluate(input1, input2, weight):
    hidden1 = sigmoid((input1 * weight[3]) + (input2 * weight[4]) + weight[5])
    hidden2 = sigmoid((input1 * weight[6]) + (input2 * weight[7]) + weight[8])
    output1 = sigmoid((hidden1 * weight[0]) + (hidden2 * weight[1]) + weight[2])
    return output1


# gradient = delta of right neuron * output of left neuron
# calculating all gradients.
def calcgradients(useInput1, useInput2, useWeight, useBias, target_output):
    output = evaluate(useInput1, useInput2, useWeight)
    output_error = output - target_output
    output_delta = output_error * sigmoidprime(output)

    hidden1 = sigmoid((useInput1 * useWeight[3]) + (useInput2 * useWeight[4]) + useWeight[5])
    hidden2 = sigmoid((useInput1 * useWeight[6]) + (useInput2 * useWeight[7]) + useWeight[8])

    hidden1_error = output_delta * useWeight[0]
    hidden2_error = output_delta * useWeight[1]

    hidden1_delta = hidden1_error * sigmoidprime(hidden1)
    hidden2_delta = hidden2_error * sigmoidprime(hidden2)

    gradient1 = output_delta * hidden1
    gradient2 = output_delta * hidden2
    gradient3 = output_delta * useBias[1]
    gradient4 = hidden1_delta * useInput1
    gradient5 = hidden1_delta * useInput2
    gradient6 = hidden1_delta * useBias[0]
    gradient7 = hidden2_delta * useInput1
    gradient8 = hidden2_delta * useInput2
    gradient9 = hidden2_delta * useBias[0]

    gradients = [
        gradient1, gradient2, gradient3, gradient4, gradient5, gradient6,
        gradient7, gradient8, gradient9
    ]
    return gradients


# Initialize the weights and biases
weight = [random.uniform(-1, 1) for _ in range(9)]
bias = [1, 1]
deltweight = [0] * 9

# Define training data
training_data = [
    (0, 0, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 0)
]

# Training loop
while error > 0.05:
    total_error = 0
    total_gradients = np.zeros(9)

    for inp1, inp2, target in training_data:
        output = evaluate(inp1, inp2, weight)
        total_error += (target - output) ** 2

        gradients = calcgradients(inp1, inp2, weight, bias, target)
        total_gradients += np.array(gradients)

    error = total_error / len(training_data)
    print(error)

    for j in range(9):
        deltweight[j] = epsilon * total_gradients[j] + alpha * deltweight[j]
        weight[j] -= deltweight[j]  # Update weights

# Print final weights
print("Final weights:", weight)
