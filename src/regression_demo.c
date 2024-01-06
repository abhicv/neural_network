#include <stdio.h>
#include <stdlib.h>

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

int main(int argc, char *argv[])
{
    Net net = {0};
    net.learnRate = 0.003f;

    AddLayer(&net, CreateLayer(0, 1, LINEAR));
    AddLayer(&net, CreateLayer(1, 10, RELU));
    AddLayer(&net, CreateLayer(10, 10, RELU));
    AddLayer(&net, CreateLayer(10, 1, LINEAR));

    float avgCost = 0.0f;

    int dataSize = sizeof(data) / sizeof(data[0]);

    int batchSize = 1;
    for (unsigned long i = 0; i < 110000; i++)
    {
        FeedForward(&net, data[i % dataSize], 1);

        float target[] = {[0] = data[i % dataSize][1]};

        float cost = ComputeMSE(net, target, 1);
        avgCost += cost;

        BackPropagate(&net, target, 1, MEAN_SQUARE_ERROR);
        ComputeGradients(&net, batchSize);
            
        if (i % batchSize == 0) 
        {
            printf("[%lu] avg. cost: %f\n", i + 1, avgCost / batchSize);
            avgCost = 0.0f;
            Update(&net);
            ZeroGradients(&net);
        }
    }
    
    for (int n = 0; n < dataSize; n++)
    {
        FeedForward(&net, data[n], 1);
        printf("output: %0.3f, expected %0.3f\n", net.layers[net.layerCount - 1].neurons[0].activation, data[n][1]);
    }

    return 0;
}
