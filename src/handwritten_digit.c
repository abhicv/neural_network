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
            printf("%0.2f ", data[y * width + x]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    // pixel data
    unsigned char *data = 0;

    FILE *dataFile = fopen("train-images.idx3-ubyte", "rb");

    if (dataFile)
    {
        fseek(dataFile, 0, SEEK_END);
        unsigned long size = ftell(dataFile);
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

    FILE *labelFile = fopen("train-labels.idx1-ubyte", "rb");

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

    // printf("data file magic number: %x\n", *((int*)data));
    // printf("data file item count: %x\n", *(int*)(data + 4));
    // printf("label file magic number: %x\n", *((int*)label));
    // printf("label file item count: %x\n", *(int*)(label + 4));

    int topology[] = {28 * 28, 100, 100, 100, 10};
    int layerCount = sizeof(topology) / sizeof(topology[0]);

    Net net = CreateNetwork(topology, layerCount, 0.001);

    // clock_t s = clock();

    for (int n = 0; n < 200; n++)
    {
        printf("epoch : %u\n", n + 1);

        for (unsigned long i = 0; i < 60000; i++)
        {
            float input[28 * 28] = {0};

            for (int n = 0; n < (28 * 28); n++)
            {
                input[n] = ((float)*(data + (4 * 4) + (i * 28 * 28) + n)) / 255.0;
            }

            FeedForward(&net, input, 28 * 28);

            float target[10] = {0};
            int index = (int)*(label + (4 * 2) + i);
            target[index] = 1.0;

            // printf("target: ");
            // for(int n = 0; n < 10; n++) printf("%0.1f ", target[n]);
            // printf("\n");

            // printf("output: ");
            // for (int n = 0; n < 10; n++)
            //     printf("%0.1f ", net.layers[net.layerCount - 1].neurons[n].activation);
            // printf("\n");

            BackPropagate(&net, target, 10);

            // printf("digit: %d\n", index);
            // PrintImage(input, 28, 28);
        }
    }
    // printf("training time: %ld cpu time\n", (clock() - s));

    WriteNetworkToFile(net, "digit.wanb");
    return 0;
}

// parameters = 93800
//  hidden_layer = [100, 100, 100], learn_rate = 0.001, epoch = 50, accuracy : 0.206
//  hidden_layer = [100, 100, 100], learn_rate = 0.001, epoch = 100, accuracy : 0.207
//  hidden_layer = [100, 100, 100], learn_rate = 0.001, epoch = 200, accuracy :

// parameters = 89610
//  hidden_layer = [100, 100], learn_rate = 0.001, epoch = 50, accuracy : 0.927
//  hidden_layer = [100, 100], learn_rate = 0.001, epoch = 100, accuracy : 0.952
//  hidden_layer = [100, 100], learn_rate = 0.001, epoch = 200, accuracy : 0.970

// parameters = 73800
//  hidden_layer = [100], LR=0.001, epoch=50, accuracy : 0.953
//  hidden_layer = [100], LR=0.001, epoch=100, accuracy : 0.962
//  hidden_layer = [100], LR=0.001, epoch=200, accuracy : 0.971
