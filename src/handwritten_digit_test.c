#include <stdio.h>

#include "nn.c"
#include "util.c"

int main(int argc, char *argv[])
{
    char *data = ReadBinaryFileIntoMemory("data/training_data/t10k-images.idx3-ubyte");
    if(!data) return 0;

    char *label = ReadBinaryFileIntoMemory("data/training_data/t10k-labels.idx1-ubyte");
    if(!label) return 0;

    Net net = LoadNetworkFromFile("digit_96.wanb");
    
    int correctPrediction = 0;
    int total = 10000;
    for (unsigned long i = 0; i < total; i++)
    {
        float input[28 * 28] = {0};

        for (int n = 0; n < (28 * 28); n++)
        {
            unsigned char c = (unsigned char)*(data + (4 * 4) + (i * 28 * 28) + n);
            input[n] = (float)c / 255.0f;
        }

        FeedForward(&net, input, 28 * 28);

        int digit = (int)(*(label + (4 * 2) + i));

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

        if(digit == prediction) correctPrediction++;
    }

    printf("correct / total : %d / %d\n", correctPrediction, total);
    printf("accuracy : %0.4f\n", (float)correctPrediction / (float)total);

    return 0;
}
