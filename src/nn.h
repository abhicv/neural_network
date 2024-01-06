#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
} Neuron;

typedef struct Layer
{
    Neuron *neurons;
    unsigned int neuronCount;
    enum ActivationFunction activationFunc;
} Layer;

typedef struct Net
{
    Layer *layers;
    unsigned int layerCount;
    unsigned int *topology;
    float learnRate;
} Net;

#endif // NN_H
