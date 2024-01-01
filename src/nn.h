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
    // float *weights;
    // float bias;
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

Neuron CreateNeuron(unsigned int neuronCountPreviousLayer);
Layer CreateLayer(unsigned int neuronCountCurrentLayer, unsigned int neuronCountPreviousLayer);
Net CreateNetwork(unsigned int *topology, unsigned int layerCount, float learnRate);

void FeedForward(Net *net, float *input, unsigned int inputCount);
void BackPropagate(Net *net, float *target, unsigned int targetCount);
float ComputeCost(Net net, float *target, unsigned int targetCount);

float ReLU(float z);
float dReLU_dz(float z);

float sigmoid(float z);
float dsigmoid_dz(float z);

void LogWeights(Net net);
void WriteNetworkToFile(Net net, const char *fileName);
Net LoadNetworkFromFile(const char *fileName);

#endif // NN_H
