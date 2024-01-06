#include <stdio.h>

#include "nn.c"
#include "util.c"

int main(int argc, char *argv[])
{
    char *data = ReadBinaryFileIntoMemory("data/training_data/t10k-images.idx3-ubyte");
    if(!data) return 0;

    char *label = ReadBinaryFileIntoMemory("data/training_data/t10k-labels.idx1-ubyte");
    if(!label) return 0;

    // char *data = ReadBinaryFileIntoMemory("data/training_data/train-images.idx3-ubyte");
    // if (!data) return 1;

    // char *label = ReadBinaryFileIntoMemory("data/training_data/train-labels.idx1-ubyte");
    // if (!label) return 1;

    Net net = LoadNetworkFromFile("digit_90.wanb");
    net.layers[0].activationFunc = LINEAR;
    net.layers[1].activationFunc = RELU;
    net.layers[2].activationFunc = RELU;
    net.layers[3].activationFunc = SOFTMAX;
    
    int correctPrediction = 0;
    int total = 10;
    for (unsigned long i = 0; i < total; i++)
    {
        float input[28 * 28] = {0};

        for (int n = 0; n < (28 * 28); n++)
        {
            input[n] = ((float)*(data + (4 * 4) + (i * 28 * 28) + n)) / 255.0f;
        }

        _FeedForward(&net, input, 28 * 28);

        int digit = (int)(*(label + (4 * 2) + i));
        // printf("digit: %d\n", digit);

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

        if(digit == prediction)
        {
            correctPrediction++;
        }
        else
        {
            printf("output: ");
            for (int n = 0; n < 10; n++)
            {
                float a = net.layers[net.layerCount - 1].neurons[n].activation;
                printf("%f ", a);
            }
            printf("\n");
            printf("target: %d, prediction: %d\n", digit, prediction);
            // PrintImage(input, 28, 28);
        }
    }

    printf("correct / total : %d / %d\n", correctPrediction, total);
    printf("accuracy : %0.4f\n", (float)correctPrediction / (float)total);

    return 0;
}
