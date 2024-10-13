#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

enum ActivationFunction
{
    LINEAR = 0,
    RELU,
    SIGMOID,
    SOFTMAX,
    TANH,
};

typedef struct Value
{
    float data;
    float grad;
} Value;

typedef struct Neuron
{
    float activation;
    float z;
    Value *weights;
    Value bias;
    float dcost_da;
    bool dead;
} Neuron;

typedef struct Layer
{
    Neuron *neurons;
    unsigned int neuronCount;
    enum ActivationFunction activationFunc;
    float dropRate;
} Layer;

typedef struct Net
{
    Layer *layers;
    unsigned int layerCount;
    float learnRate;
} Net;

extern __declspec(dllexport) void FeedForward(Net *net, float *input, unsigned int inputCount);
extern __declspec(dllexport) Net LoadNetworkFromFile(const char *fileName);

#endif // NN_H
