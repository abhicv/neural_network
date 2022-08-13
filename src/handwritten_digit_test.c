#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "nn.c"

void PrintImage(float *data, int width, int height)
{
    printf("\n");
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if(data[y * width + x] > 0.0)
                // printf("\033[0;33m%0.2f ", data[y * width + x]);
                printf("#");
            else
                printf(" ");
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    // pixel data
    unsigned char *data = 0;

    FILE *dataFile = fopen("t10k-images.idx3-ubyte", "rb");

    if (dataFile)
    {
        fseek(dataFile, 0, SEEK_END);
        unsigned int size = ftell(dataFile);
        fseek(dataFile, 0, SEEK_SET);
        data = (char *)malloc(size);
        fread(data, 1, size, dataFile);
        fclose(dataFile);
    }
    else
    {
        printf("error: failed to open data file !\n");
        return 1;
    }

    FILE *labelFile = fopen("t10k-labels.idx1-ubyte", "rb");
    
    unsigned char *label = 0;
    if (labelFile)
    {
        fseek(labelFile, 0, SEEK_END);
        unsigned int size = ftell(labelFile);
        fseek(labelFile, 0, SEEK_SET);
        label = (char *)malloc(size);
        fread(label, 1, size, labelFile);
        fclose(labelFile);
    }
    else
    {
        printf("error: failed to open label file !\n");
        return 1;
    }

    // printf("data file magic number: %x\n", *((int *)data));
    // printf("data file item count: %x\n", *(int *)(data + 4));
    // printf("label file magic number: %x\n", *((int *)label));
    // printf("label file item count: %x\n", *(int *)(label + 4));

    Net net = LoadNetworkFromFile("digit.wanb");
    // printf("%d\n", net.layerCount);

    int correctPrediction = 0;
    int total = 10000;
    for (unsigned long i = 0; i < total; i++)
    {
        float input[28 * 28] = {0};

        for (int n = 0; n < (28 * 28); n++)
        {
            input[n] = ((float)*(data + (4 * 4) + (i * 28 * 28) + n)) / 255.0;
        }

        FeedForward(&net, input, 28 * 28);

        int digit = (int)*(label + (4 * 2) + i);

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
            // printf("target: %d, prediction: %d\n", digit, prediction);
            // PrintImage(input, 28, 28);
        }
    }

    printf("accuracy : %0.3f\n", (float)correctPrediction / (float)total);
    return 0;
}
