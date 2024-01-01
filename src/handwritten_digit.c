#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "nn.c"
#include "util.c"

int main(int argc, char *argv[])
{
    char *data = ReadBinaryFileIntoMemory("data/training_data/train-images.idx3-ubyte");
    if (!data) return 1;

    char *label = ReadBinaryFileIntoMemory("data/training_data/train-labels.idx1-ubyte");
    if (!label) return 1;

    int topology[] = {28 * 28, 512, 100, 10};
    int layerCount = sizeof(topology) / sizeof(topology[0]);

    Net net = CreateNetwork(topology, layerCount, 0.001f);

    for (int n = 0; n < 400; n++)
    {
        printf("epoch : %u, ", n + 1);

        float avgCost = 0.0f;
        int correctPrediction = 0;

        for (unsigned long i = 0; i < 60000; i++)
        {
            float input[28 * 28] = {0};

            for (int j = 0; j < (28 * 28); j++)
            {
                input[j] = ((float)*(data + (4 * 4) + (i * 28 * 28) + j)) / 255.0;
            }

            FeedForward(&net, input, 28 * 28);

            int index = (int)*(label + (4 * 2) + i);

            float target[10] = {0};
            target[index] = 1.0f;

            BackPropagate(&net, target, 10);

            avgCost += ComputeCost(net, target, 10);

            float prob = 0;
            int prediction = -1;
            for (int n = 0; n < 10; n++)
            {
                if(prob < net.layers[net.layerCount - 1].neurons[n].activation)
                {
                    prob = net.layers[net.layerCount - 1].neurons[n].activation;
                    prediction = n;
                }
            }

            if(index == prediction) correctPrediction++;
            
            // printf("target: ");
            // for(int n = 0; n < 10; n++) printf("%0.1f ", target[n]);
            // printf("\n");

            // printf("output: ");
            // for (int n = 0; n < 10; n++)
            //     printf("%0.1f ", net.layers[net.layerCount - 1].neurons[n].activation);
            // printf("\n");

            // printf("digit: %d\n", index);
            // PrintImage(input, 28, 28);
        }

        printf("Cost: %f, (%d / %d), accuracy: %f\n", avgCost / 60000.0f, correctPrediction, 60000, (float)correctPrediction * 100.0f / 60000.0f);
    }

    WriteNetworkToFile(net, "digit.wanb");
    return 0;
}
