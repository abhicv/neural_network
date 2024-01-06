#include <stdio.h>

#include "nn.c"

float data[4][3] = {
    [0] = {0.0, 0.0, 1.0},
    [1] = {1.0, 0.0, 0.0},
    [2] = {0.0, 1.0, 0.0},
    [3] = {1.0, 1.0, 1.0}
};

int main(int argc, char *argv[])
{
    Net net = {0};
    net.learnRate = 0.1f;

    int dataSize = sizeof(data) / sizeof(data[0]);

    AddLayer(&net, CreateLayer(0, 2, LINEAR));
    AddLayer(&net, CreateLayer(2, 3, RELU));
    AddLayer(&net, CreateLayer(3, 1, LINEAR));

    for(unsigned long i = 0; i < 250; i++)
    {
        FeedForward(&net, data[i % dataSize], 2);

        float target[] = { data[i % dataSize][2] };

        float cost = ComputeMSE(net, target, 1);
        printf("[%lu] cost: %f\n", i + 1, cost);

        BackPropagate(&net, target, 1, MEAN_SQUARE_ERROR);
        ComputeGradients(&net, 1);
        Update(&net);
        ZeroGradients(&net);
    }

    for (int n = 0; n < dataSize; n++)
    {
        FeedForward(&net, data[n], 2);
        printf("output: %0.3f, expected %0.3f\n", net.layers[net.layerCount - 1].neurons[0].activation, data[n][2]);
    }

    return 0;
}
