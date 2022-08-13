#include "nn.h"

Neuron CreateNeuron(unsigned int neuronCountPreviousLayer)
{
    Neuron neuron = {0};

    if (neuronCountPreviousLayer > 0)
    {
        neuron.weights = (float *)malloc(sizeof(float) * neuronCountPreviousLayer);

        for (int i = 0; i < neuronCountPreviousLayer; i++)
        {
            neuron.weights[i] = (float)(rand() % 5 + 1) * 0.001;
        }
    }

    neuron.bias = (float)(rand() % 5 + 1) * 0.001;

    return neuron;
}

Layer CreateLayer(unsigned int neuronCountCurrentLayer, unsigned int neuronCountPreviousLayer)
{
    Layer layer = {0};
    layer.neuronCount = neuronCountCurrentLayer;

    layer.neurons = (Neuron *)malloc(sizeof(Neuron) * layer.neuronCount);

    for (int i = 0; i < layer.neuronCount; i++)
    {
        layer.neurons[i] = CreateNeuron(neuronCountPreviousLayer);
    }

    return layer;
}

Net CreateNetwork(unsigned int *topology, unsigned int layerCount, float learnRate)
{
    Net net = {0};
    net.layerCount = layerCount;
    net.topology = topology;
    net.layers = (Layer *)malloc(sizeof(Layer) * net.layerCount);
    net.learnRate = learnRate;

    for (int i = 0; i < net.layerCount; i++)
    {
        if (i == 0)
        {
            net.layers[i] = CreateLayer(topology[i], 0);
        }
        else
        {
            net.layers[i] = CreateLayer(topology[i], topology[i - 1]);
        }
    }
    return net;
}

float sigmoid(float z)
{
    return exp(z) / (1 + exp(z));
}

float dsigmoid_dz(float z)
{
    return sigmoid(z) * (1 - sigmoid(z));
}

float ReLU(float z)
{
    return z > 0 ? z : 0;
}

float dReLU_dz(float z)
{
    return z > 0 ? 1 : 0;
}

void FeedForward(Net *net, float *input, unsigned int inputCount)
{
    if (inputCount != net->layers[0].neuronCount)
    {
        printf("error: no. of inputs != no.of input layer neurons\n");
        return;
    }

    // input layer
    for (int i = 0; i < inputCount; i++)
    {
        net->layers[0].neurons[i].activation = input[i];
    }

    int outputLayerIndex = net->layerCount - 1;

    // hidden layer and output layer
    for (int i = 1; i < net->layerCount; i++)
    {
        for (int j = 0; j < net->layers[i].neuronCount; j++)
        {
            float weightedSum = 0;

            for (int k = 0; k < net->layers[i - 1].neuronCount; k++)
            {
                weightedSum += net->layers[i].neurons[j].weights[k] * net->layers[i - 1].neurons[k].activation;
            }

            weightedSum += net->layers[i].neurons[j].bias;
            net->layers[i].neurons[j].z = weightedSum;

            if (i == outputLayerIndex)
            {
                net->layers[i].neurons[j].activation = weightedSum;
            }
            else
            {
                net->layers[i].neurons[j].activation = sigmoid(weightedSum);
            }
        }
    }
}

void BackPropagate(Net *net, float *target, unsigned int targetCount)
{
    if (net->layers[net->layerCount - 1].neuronCount != targetCount)
    {
        printf("error: no.of neurons in output layer != target count\n");
        return;
    }

    for (int i = net->layerCount - 1; i > 0; i--)
    {
        for (int k = 0; k < net->layers[i].neuronCount; k++)
        {
            net->layers[i].neurons[k].dcost_da = 0;

            if (i == net->layerCount - 1)
            {
                net->layers[i].neurons[k].dcost_da = 2 * (net->layers[i].neurons[k].activation - target[k]);
            }
            else
            {
                for (int j = 0; j < net->layers[i + 1].neuronCount; j++)
                {
                    float da_dz = 0.0;
                    if ((i + 1) == net->layerCount - 1)
                    {
                        da_dz = 1.0;
                    }
                    else
                    {
                        da_dz = dsigmoid_dz(net->layers[i + 1].neurons[j].z);
                    }

                    net->layers[i].neurons[k].dcost_da += net->layers[i + 1].neurons[j].weights[k] * da_dz * net->layers[i + 1].neurons[j].dcost_da;
                }
            }
        }
    }

    for (int i = 1; i < net->layerCount; i++)
    {
        for (int j = 0; j < net->layers[i].neuronCount; j++)
        {
            float dcost_dz = 0.0;
            if (i == net->layerCount - 1)
            {
                dcost_dz = net->layers[i].neurons[j].dcost_da;
            }
            else
            {
                dcost_dz = dsigmoid_dz(net->layers[i].neurons[j].z) * net->layers[i].neurons[j].dcost_da;
            }

            for (int k = 0; k < net->layers[i - 1].neuronCount; k++)
            {
                net->layers[i].neurons[j].weights[k] -= net->learnRate * net->layers[i - 1].neurons[k].activation * dcost_dz;
            }

            net->layers[i].neurons[j].bias -= net->learnRate * dcost_dz;
        }
    }
}

float ComputeCost(Net net, float *target, unsigned int targetCount)
{
    if (net.layers[net.layerCount - 1].neuronCount != targetCount)
    {
        printf("error: no.of neurons in output layer != target count\n");
        return 0;
    }

    float cost = 0;

    for (int i = 0; i < targetCount; i++)
    {
        cost += (target[i] - net.layers[net.layerCount - 1].neurons[i].activation) * (target[i] - net.layers[net.layerCount - 1].neurons[i].activation);
    }

    return cost;
}

void LogWeights(Net net)
{
    for (int i = 1; i < net.layerCount; i++)
    {
        for (int j = 0; j < net.layers[i].neuronCount; j++)
        {
            for (int k = 0; k < net.layers[i - 1].neuronCount; k++)
            {
                printf("layer: [%d] from: [%d] to: [%d] weight: %0.3f\n", i, k + 1, j + 1, net.layers[i].neurons[j].weights[k]);
            }
        }
    }
}

void WriteNetworkToFile(Net net, const char *fileName)
{
    FILE *out = fopen(fileName, "w");

    if (!out)
    {
        printf("error: failed to create output file '%s'\n", fileName);
        return;
    }

    fprintf(out, "l: %d\n", net.layerCount);
    fprintf(out, "n:");
    for (int n = 0; n < net.layerCount; n++)
    {
        fprintf(out, " %d", net.layers[n].neuronCount);
    }
    fprintf(out, "\n");

    for (int i = 1; i < net.layerCount; i++)
    {
        for (int j = 0; j < net.layers[i].neuronCount; j++)
        {
            for (int k = 0; k < net.layers[i - 1].neuronCount; k++)
            {
                fprintf(out, "w: %0.6f\n", net.layers[i].neurons[j].weights[k]);
            }
            fprintf(out, "b: %0.6f\n", net.layers[i].neurons[j].bias);
        }
    }
    fclose(out);
}

Net LoadNetworkFromFile(const char *fileName)
{
    Net net = {0};
    unsigned int layerCount = 0;

    FILE *in = fopen(fileName, "r");

    if (!in)
    {
        printf("error: failed to open file '%s'\n", fileName);
        return net;
    }

    fscanf(in, "l: %u\n", &layerCount);
    unsigned int *topology = malloc(sizeof(unsigned int) * layerCount);

    fscanf(in, "n:");
    for (int n = 0; n < layerCount; n++)
    {
        fscanf(in, " %d", topology + n);
    }
    fscanf(in, "\n");

    net = CreateNetwork(topology, layerCount, 0.001);

    for (int i = 1; i < net.layerCount; i++)
    {
        for (int j = 0; j < net.layers[i].neuronCount; j++)
        {
            for (int k = 0; k < net.layers[i - 1].neuronCount; k++)
            {
                fscanf(in, "w: %f\n", &net.layers[i].neurons[j].weights[k]);
            }
            fscanf(in, "b: %f\n", &net.layers[i].neurons[j].bias);
        }
    }

    fclose(in);

    free(topology);

    return net;
}
