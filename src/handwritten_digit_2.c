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
    net.layerCount = 3;
    net.layers = (Layer *)malloc(sizeof(Layer) * net.layerCount);
    net.learnRate = 0.005f;

    net.layers[0] = _CreateLayer(0, inputCount, LINEAR);
    net.layers[1] = _CreateLayer(inputCount, 100, RELU);
    net.layers[2] = _CreateLayer(100, 10, LINEAR);
    // net.layers[3] = _CreateLayer(50, 10, LINEAR);

    for (int n = 0; n < 200; n++)
    {
        printf("epoch : %u, ", n + 1);

        float avgCost = 0.0f;
        int correctPrediction = 0;

        for (unsigned long i = 0; i < 60000; i++)
        {
            float input[28 * 28] = {0};

            for (int j = 0; j < (28 * 28); j++)
            {
                input[j] = ((float)*(data + (4 * 4) + (i * 28 * 28) + j)) / 255.0f;
            }

            _FeedForward(&net, input, 28 * 28);

            int index = (int)*(label + (4 * 2) + i);

            float target[10] = {0};
            target[index] = 1.0f;

            _BackPropagate(&net, target, 10, MEAN_SQUARE_ERROR);

            avgCost += ComputeCost(net, target, 10);
            // avgCost += ComputeCrossEntropyLoss(net, target, 10);

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

            // if(i % 10000 == 0)
            if(0)
            {
                printf("digit: %d\n", index);
                // PrintImage(input, 28, 28);

                printf("target: ");
                for(int n = 0; n < 10; n++) printf("%0.4f ", target[n]);
                printf("\n");

                printf("output: ");
                float sum = 0.0f;
                float exps[10] = {0};
                float expSums = 0.0f;

                for (int n = 0; n < 10; n++)
                {
                    float a = net.layers[net.layerCount - 1].neurons[n].activation;
                    printf("%0.4f ", a);
                    sum += a;
                    exps[n] = expf(a);
                    expSums += exps[n];
                }
                printf("\n");

                // normalized output

                printf("norm: ");
                
                for (int n = 0; n < 10; n++)
                {
                    printf("%0.4f ", net.layers[net.layerCount - 1].neurons[n].activation / sum);
                }
                printf("\n");

                printf("softmax: ");
                
                for (int n = 0; n < 10; n++)
                {
                    printf("%0.4f ", exps[n] / expSums);
                }
                printf("\n");
            }            
        }

        printf("Cost: %f, (%d / %d), accuracy: %f\n", avgCost / 60000.0f, correctPrediction, 60000, (float)correctPrediction * 100.0f / 60000.0f);
    }

    WriteNetworkToFile(net, "digit.wanb");
    return 0;
}
