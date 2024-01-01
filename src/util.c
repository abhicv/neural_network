#include <stdio.h>

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

char *ReadBinaryFileIntoMemory(const char* fileName)
{
    char *data = 0;

    FILE *file = fopen(fileName, "rb");

    if (!file)
    {
        printf("error opening binary file : '%s'\n", fileName);
        return 0;
    }

    fseek(file, 0, SEEK_END);
    unsigned int size = ftell(file);
    fseek(file, 0, SEEK_SET);
    data = (char *)malloc(size);
    fread(data, 1, size, file);
    fclose(file);

    return data;
}
