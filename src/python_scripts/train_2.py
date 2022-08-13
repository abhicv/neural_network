import nn
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import math

def function(x):
    return math.cos(x)

size = 20
input = []
target = []

angle = 0
step = (2 * math.pi) / size

for n in range(0, size):
    input.append([angle])
    target.append([function(angle)])
    angle += step

plt.plot(input, target, label='true', color='r')
plt.ion()
plt.show()

topology = [1, 3, 1]
net = nn.create_network(topology, 0.01)

# training
for e in  range(0, 10000):
    prediction = []
    for n in range(0, len(input)):
        nn.feedforward(net, input[n])
        nn.backpropagate(net, target[n])
        prediction.append(net.layers[net.layer_count-1].neurons[0].activation)
    # print(e)
    plt.clf()
    plt.plot(input, target, label='true', color='r')
    plt.scatter(input, prediction, label='prediction' , color='b')
    plt.legend()
    plt.pause(1e-6)

# testing
# for n in range(0, size):
#     net = nn.feedforward(net, [angle])
