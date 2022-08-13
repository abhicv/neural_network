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
    int topology[] = {2, 3, 1};
    int layerCount = sizeof(topology) / sizeof(topology[0]);

    Net net = CreateNetwork(topology, layerCount, 0.1);

    clock_t s = clock();
    
    for(unsigned long i = 0; i < 50000; i++)
    {
        FeedForward(&net, data[i % LEN], 2);
        float target[] = { data[i % LEN][2] };
        BackPropagate(&net, target, 1);
    }

    printf("training time: %ld cpu time\n", (clock() - s));

    for (int n = 0; n < LEN; n++)
    {
        FeedForward(&net, data[n], 2);
        printf("output: %0.3f, expected %0.3f\n", net.layers[net.layerCount - 1].neurons[0].activation, data[n][2]);
    }

    return 0;
}
