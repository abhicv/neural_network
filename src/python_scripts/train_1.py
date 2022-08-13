import nn
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import math

topology = [1, 3, 1]
net = nn.create_network(topology, 0.01)

input = [
    [0.0000],
    [0.6283],
    [1.2566],
    [1.8850],
    [2.5133],
    [3.1416],
    [3.7699],
    [4.3982],
    [5.0265],
    [5.6549],
    [6.2832]
]

target = [ 
    [0.0000],
    [0.5878],
    [0.9511],
    [0.9511],
    [0.5878],
    [0.0000],
    [-0.5878],
    [-0.9511],
    [-0.9511],
    [-0.5878],
    [0.0000]
]

# training
for e in  range(0, 20000):
    for n in range(0, len(input)):
        nn.feedforward(net, input[n])
        nn.backpropagate(net, target[n])

prediction = []
true = []
angles = []

division = 80
step = (2 * math.pi) / division
angle = 0;

# testing
for n in range(0, division):
    i = [angle]
    nn.feedforward(net, i)
    prediction.append(net.layers[net.layer_count-1].neurons[0].activation)
    true.append(math.sin(angle))
    angles.append(angle)
    angle += step
    # print("sin " + str(i) + " = " + str(net.layers[net.layer_count-1].neurons[0].activation))

plt.plot(angles, true, label='true', color='r')
plt.scatter(angles, prediction, label='prediction' , color='g')
plt.legend()
plt.show()