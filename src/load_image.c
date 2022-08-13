#include <stdio.h>
#include <stdlib.h>

void PrintImage(unsigned char *data, int width, int height)
{
    printf("\n");           
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            printf("%0.2f ", (float)data[y * width + x] / (float)255);           
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
    
    if(dataFile)
    {
        fseek(dataFile, 0, SEEK_END);
        unsigned int size = ftell(dataFile);
        fseek(dataFile, 0, SEEK_SET);
        data = (char*)malloc(size);
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
    if(labelFile)
    {
        fseek(labelFile, 0, SEEK_END);
        unsigned int size = ftell(labelFile);
        fseek(labelFile, 0, SEEK_SET);
        label = (char*)malloc(size);
        fread(label, 1, size, labelFile);
        fclose(labelFile);
    }
    else
    {
        printf("error: failed to open label file !\n");
        return 1;
    }    

    printf("data file magic number: %x\n", *((int*)data));
    printf("data file item count: %x\n", *(int*)(data + 4));
    printf("label file magic number: %x\n", *((int*)label));
    printf("label file item count: %x\n", *(int*)(label + 4));

    for(int n = 0; n < 5; n++)
    {
        printf("digit: %d\n", (int)*(label + (4*2) + n));
        PrintImage(data + (4*4) + (n * 28*28), 28, 28);
    }

    return 0;
}