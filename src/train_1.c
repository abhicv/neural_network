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
    int topology[] = {1, 3, 1};
    int layerCount = sizeof(topology) / sizeof(topology[0]);

    Net net = CreateNetwork(topology, layerCount, 0.01);

    clock_t s = clock();

    for (unsigned long i = 0; i < 1000000; i++)
    {
        FeedForward(&net, data[i % LEN], 1);        
        float target[] = {[0] = data[i % LEN][1]};
        BackPropagate(&net, target, 1);
        float avgCost = ComputeCost(net, target, 1);
        printf("[%lu] cost: %f\n", i + 1, avgCost);
    }
    
    printf("training time: %ld cpu time\n", (clock() - s));

    for (int n = 0; n < LEN; n++)
    {
        FeedForward(&net, data[n], 1);
        printf("output: %0.3f, expected %0.3f\n", net.layers[net.layerCount - 1].neurons[0].activation, data[n][1]);
    }

    // WriteNetworkToFile(net, "out.wanb");

    // Net n = LoadNetworkFromFile("out.wanb");

    // for(float i = 0; i < (2 * 3.14159); i += (2 * 3.14159) / 20)
    // {
    //     float input[] = {[0] = sinf(i)};
    //     FeedForward(&net, input, 1);
    //     printf("output: %0.4f, expected: %0.4f,\n", net.layers[net.layerCount - 1].neurons[0].activation, input[0]);
    // }

    return 0;
}
