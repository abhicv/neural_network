#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "nn.c"

float data[11][2] = {
    [0] = {0.0000, 0.0000},
    [1] = {0.6283, 0.5878},
    [2] = {1.2566, 0.9511},
    [3] = {1.8850, 0.9511},
    [4] = {2.5133, 0.5878},
    [5] = {3.1416, 0.0000},
    [6] = {3.7699, -0.5878},
    [7] = {4.3982, -0.9511},
    [8] = {5.0265, -0.9511},
    [9] = {5.6549, -0.5878},
    [10] = {6.2832, 0.0000},
};

#define LEN 11

int main(int argc, char *argv[])
{
    Net net = {0};
    net.learnRate = 0.001f;

    AddLayer(&net, _CreateLayer(0, 1, LINEAR));
    AddLayer(&net, _CreateLayer(1, 10, RELU));
    AddLayer(&net, _CreateLayer(10, 10, RELU));
    AddLayer(&net, _CreateLayer(10, 1, LINEAR));

    float avgCost = 0.0f;

    int batchSize = 1;
    for (unsigned long i = 0; i < 110000; i++)
    {
        _FeedForward(&net, data[i % LEN], 1);

        float target[] = {[0] = data[i % LEN][1]};

        float cost = ComputeCost(net, target, 1);
        avgCost += cost;

        _BackPropagate(&net, target, 1, MEAN_SQUARE_ERROR);
        _ComputeGradients(&net, batchSize);
        
        if (i % batchSize == 0) 
        {
            printf("[%lu] avg. cost: %f\n", i + 1, avgCost / 11.0f);
            avgCost = 0.0f;
            _Update(&net);
            _ZeroGradients(&net);
        }
    }
    
    for (int n = 0; n < LEN; n++)
    {
        _FeedForward(&net, data[n], 1);
        printf("output: %0.3f, expected %0.3f\n", net.layers[net.layerCount - 1].neurons[0].activation, data[n][1]);
    }

    // WriteNetworkToFile(net, "out.wanb");

    // Net n = LoadNetworkFromFile("out.wanb");

    // for(float i = 0; i < (2 * 3.14159); i += (2 * 3.14159) / 30)
    // {
    //     float input[] = {[0] = sinf(i)};
    //     _FeedForward(&net, input, 1);
    //     printf("output: %0.4f, expected: %0.4f,\n", net.layers[net.layerCount - 1].neurons[0].activation, input[0]);
    // }

    return 0;
}
