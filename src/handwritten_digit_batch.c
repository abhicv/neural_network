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

    int inputCount = 28 * 28;
    
    Net net = {0};
    net.learnRate = 0.001f;

    AddLayer(&net, _CreateLayer(0, inputCount, LINEAR));
    AddLayer(&net, _CreateLayer(inputCount, 100, RELU));
    AddLayer(&net, _CreateLayer(100, 100, RELU));
    AddLayer(&net, _CreateLayer(100, 10, SOFTMAX));

    int batchSize = 1;
    for (int n = 0; n < 1000; n++)
    {
        printf("epoch : %u, ", n + 1);

        float avgCost = 0.0f;
        int correctPrediction = 0;

        int totalSample = 60000;
        for (unsigned long i = 0; i < totalSample; i++)
        {
            float input[28 * 28] = {0};

            for (int j = 0; j < (28 * 28); j++)
            {
                input[j] = ((float)*(data + (4 * 4) + (i * 28 * 28) + j)) / 255.0f;
            }

            int index = (int)*(label + (4 * 2) + i);
            
            _FeedForward(&net, input, 28 * 28);

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

            float target[10] = {0};
            target[index] = 1.0f;

            // float cost = ComputeCost(net, target, 10);
            float cost = ComputeCrossEntropyLoss(net, target, 10);
            avgCost += cost;

            _BackPropagate(&net, target, 10, CROSS_ENTROPY_LOSS);
            _ComputeGradients(&net, batchSize);

            if(i % batchSize == 0)
            {
                _Update(&net);
                _ZeroGradients(&net);
            }

            if(0)
            {
                printf("\n");
                printf("digit: %d\n", index);
                printf("cost: %f\n", cost);
                // PrintImage(input, 28, 28);

                printf("target: ");
                for(int n = 0; n < 10; n++) printf("%f ", target[n]);
                printf("\n");

                printf("output: ");
                for (int n = 0; n < 10; n++)
                {
                    float a = net.layers[net.layerCount - 1].neurons[n].activation;
                    printf("%f ", a);
                }
                printf("\n");
            }
        }

        printf("Cost: %f, (%d / %d), accuracy: %f %%\n", avgCost / totalSample, correctPrediction, totalSample, (float)correctPrediction * 100.0f / totalSample);
        WriteNetworkToFile(net, "digit.wanb");
    }

    return 0;
}
