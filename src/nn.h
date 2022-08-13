#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct Neuron Neuron;
struct Neuron
{
    float activation;
    float z;
    float *weights;
    float bias;
    float dcost_da;
};

typedef struct Layer Layer;
struct Layer
{
    Neuron *neurons;
    unsigned int neuronCount;
};

typedef struct Net Net;
struct Net
{
    Layer *layers;
    unsigned int layerCount;
    unsigned int *topology;
    float learnRate;
    unsigned int outputLayerActivation;
    unsigned int hiddenLayerActivation;
};

Neuron CreateNeuron(unsigned int neuronCountPreviousLayer);
Layer CreateLayer(unsigned int neuronCountCurrentLayer, unsigned int neuronCountPreviousLayer);
Net CreateNetwork(unsigned int *topology, unsigned int layerCount, float learnRate);

float sigmoid(float z);
float ReLU(float z);

float dReLU_dz(float z);
float dsigmoid_dz(float z);

void FeedForward(Net *net, float *input, unsigned int inputCount);
void BackPropagate(Net *net, float *target, unsigned int targetCount);
float ComputeCost(Net net, float *target, unsigned int targetCount);

void LogWeights(Net net);
void WriteNetworkToFile(Net net, const char *fileName);
Net LoadNetworkFromFile(const char *fileName);

#endif // NN_H
