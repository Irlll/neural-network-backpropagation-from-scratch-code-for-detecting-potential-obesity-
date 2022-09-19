import numpy as np
import math

# input
input = np.array(
    [
        [2, 2, 3, 3],
        [1, 2, 3, 3],
        [2, 1, 4, 1],
        [1, 1, 4, 2],
        [1, 1, 3, 4],
        [1, 1, 3, 1],
        [1, 1, 3, 2],
        [1, 2, 1, 2],
        [1, 2, 3, 1],
        [1, 1, 4, 1],
        [2, 1, 3, 3],
        [2, 1, 1, 2],
        [1, 1, 4, 2],
        [1, 2, 2, 3],
        [2, 1, 4, 2],
        [2, 1, 3, 1],
        [2, 2, 3, 2],
        [2, 1, 1, 1],
        [1, 2, 3, 4],
        [1, 2, 3, 2],
    ]
)

# output
target = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1])

# fungsi sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


sig = np.vectorize(sigmoid)

# inisialisasi bobot dan bias pada hidden layer
w_hidden = np.array(
    [[0.05, 0.3, 0.55], [0.1, 0.35, 0.6], [0.15, 0.4, 0.65], [0.2, 0.45, 0.7]]
)

b_hidden = np.array([[0.25, 0.5, 0.75]])

# inisialisasi bobot dan bias pada output layer
w_output = np.array(
    [
        [0.8],
        [0.85],
        [0.9],
    ]
)

b_output = np.array([[0.95]])

lr = 0.5
epochs = 200
a = 1

for epoch in range(epochs):
    print("epoch    : ", a)
    prediction = np.zeros(target.shape)

    for idx, inp in enumerate(input):
        # batch

        # feedforward
        o_hidden = np.matmul(input[idx], w_hidden) + b_hidden
        o_hidden = sig(o_hidden)

        o_output = np.matmul(o_hidden, w_output) + b_output
        o_output = sig(o_output)

        error = (target[idx] - o_output) ** 2
        prediction[idx] = o_output.round()

        print(
            "Input    :",
            input[idx],
            "   || target    :",
            target[idx],
            "   || prediction     :",
            math.trunc(prediction[idx]),
            "   || error   :",
            error,
        )

        # backpropagation
        w_output = w_output - lr * (-2 * (target[idx] - o_output)) * (
            o_output * (1 - o_output)
        ) * (o_hidden.T)
        b_output = b_output - lr * (-2 * (target[idx] - o_output)) * (
            o_output * (1 - o_output)
        ) * (1)

        w_hidden = (
            w_hidden
            - lr
            * ((-2 * (target[idx] - o_output)) * (o_output * (1 - o_output)))
            * (w_output.T)
            * (o_hidden * (1 - o_hidden))
            * input[idx][np.newaxis].T
        )

        b_hidden = b_hidden - lr * (
            (-2 * (target[idx] - o_output)) * (o_output * (1 - o_output))
        ) * (w_output.T) * (o_hidden * (1 - o_hidden)) * (1)

    a = a + 1
    print("---\n")
