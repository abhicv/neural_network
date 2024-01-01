#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "nn.c"

// Data set for training
float data[4][3] = {
    [0] = {0.0, 0.0, 1.0},
    [1] = {1.0, 0.0, 0.0},
    [2] = {0.0, 1.0, 0.0},
    [3] = {1.0, 1.0, 1.0}
};

#define LEN 4

int main(int argc, char *argv[])
{
    Net net = {0};
    net.learnRate = 0.01f;

    AddLayer(&net, _CreateLayer(0, 2, LINEAR));
    AddLayer(&net, _CreateLayer(2, 3, RELU));
    AddLayer(&net, _CreateLayer(3, 1, LINEAR));

    for(unsigned long i = 0; i < 3000; i++)
    {
        _FeedForward(&net, data[i % LEN], 2);

        float target[] = { data[i % LEN][2] };

        float cost = ComputeCost(net, target, 1);
        printf("[%lu] cost: %f\n", i + 1, cost);

        _BackPropagate(&net, target, 1, MEAN_SQUARE_ERROR);
        _ComputeGradients(&net, 1);
        _Update(&net);
        _ZeroGradients(&net);
    }

    for (int n = 0; n < LEN; n++)
    {
        _FeedForward(&net, data[n], 2);
        printf("output: %0.3f, expected %0.3f\n", net.layers[net.layerCount - 1].neurons[0].activation, data[n][2]);
    }

    return 0;
}
