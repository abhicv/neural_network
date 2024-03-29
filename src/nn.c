#include "nn.h"
#include "math_functions.c"

// random float between -1.0 and 1.0
float RandomNorm()
{
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

Neuron CreateNeuron(unsigned int neuronCountPreviousLayer)
{
    Neuron neuron = {0};
    neuron.dead = false;

    if (neuronCountPreviousLayer > 0)
    {
        neuron.weights = (Value *)malloc(sizeof(Value) * neuronCountPreviousLayer);

        for (int i = 0; i < neuronCountPreviousLayer; i++)
        {
            neuron.weights[i].data = RandomNorm();
            neuron.weights[i].grad = 0.0f;
        }
    }

    neuron.bias.data = RandomNorm();
    neuron.bias.grad = 0.0f;

    return neuron;
}

Layer CreateLayer(unsigned int inputCount, unsigned int outputCount, enum ActivationFunction activationFunc)
{
    Layer layer = {0};
    layer.neuronCount = outputCount;
    layer.activationFunc = activationFunc;
    layer.dropRate = 0.5f;
    
    layer.neurons = (Neuron *)malloc(sizeof(Neuron) * layer.neuronCount);

    for (int i = 0; i < layer.neuronCount; i++)
    {
        layer.neurons[i] = CreateNeuron(inputCount);
    }

    return layer;
}

void AddLayer(Net *net, Layer layer)
{
    if(net->layerCount == 0)
    {
        net->layerCount++;
        net->layers = (Layer *)malloc(sizeof(Layer) * net->layerCount);
    }
    else
    {
        net->layerCount++;
        net->layers = (Layer *)realloc(net->layers, sizeof(Layer) * net->layerCount);
    }
    net->layers[net->layerCount - 1] = layer;
}

typedef float (*OneInputOneOuputFuncPtr) (float);

OneInputOneOuputFuncPtr GetActivationFunction(enum ActivationFunction activationFunc)
{
    switch (activationFunc)
    {
    case LINEAR: return linear;
    case SIGMOID: return sigmoid;
    case TANH: return tanhf;
    case RELU: return ReLU;
    case SOFTMAX: return linear; //NOTE: softmax is computed afterwards
    default: return linear;
    }
}

OneInputOneOuputFuncPtr GetActivationDerivativeFunction(enum ActivationFunction activationFunc)
{
    switch (activationFunc)
    {
    case LINEAR: return dlinear_dz;
    case SIGMOID: return dsigmoid_dz;
    case TANH: return dtanh_dz;
    case RELU: return dReLU_dz;
    default: return dlinear_dz;
    }
}

char *ActivationFunctionToString(enum ActivationFunction activationFunc)
{
    switch (activationFunc)
    {
    case LINEAR: return "LINEAR";
    case SIGMOID: return "SIGMOID";
    case TANH: return "TANH";
    case RELU: return "RELU";
    case SOFTMAX: return "SOFTMAX";
    default: return "LINEAR";
    }
}

enum ActivationFunction StringToActivationFunction(char *name)
{
    if(!strcmp(name, ActivationFunctionToString(LINEAR))) return LINEAR;
    else if(!strcmp(name, ActivationFunctionToString(SIGMOID))) return SIGMOID;
    else if(!strcmp(name, ActivationFunctionToString(TANH))) return TANH;
    else if(!strcmp(name, ActivationFunctionToString(RELU))) return RELU;
    else if(!strcmp(name, ActivationFunctionToString(SOFTMAX))) return SOFTMAX;
    else return LINEAR;
}

void FeedForward(Net *net, float *input, unsigned int inputCount)
{
    if (inputCount != net->layers[0].neuronCount)
    {
        printf("error: no. of inputs != no.of input layer neurons\n");
        return;
    }

    if(net->layerCount == 0)
    {
        printf("error: add layers to the network for feedforwading\n");
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
            // net->layers[i].neurons[j].dead = false;

            // float dropRate = net->layers[i].dropRate;
            // float throw = ((float)rand() / (float)RAND_MAX);

            // if(throw < dropRate && i < (net->layerCount - 1))
            // {
            //     net->layers[i].neurons[j].dead = true;
            //     net->layers[i].neurons[j].z = 0.0f;
            //     net->layers[i].neurons[j].activation = 0.0f;
            //     continue;
            // }

            float weightedSum = 0;
            for (int k = 0; k < net->layers[i - 1].neuronCount; k++)
            {
                weightedSum += net->layers[i].neurons[j].weights[k].data * net->layers[i - 1].neurons[k].activation;
            }
            weightedSum += net->layers[i].neurons[j].bias.data;
            net->layers[i].neurons[j].z = weightedSum;
            OneInputOneOuputFuncPtr function = GetActivationFunction(net->layers[i].activationFunc);
            float activation = function(weightedSum);
            net->layers[i].neurons[j].activation = activation;
        }
    }

    // softmax
    if(net->layers[outputLayerIndex].activationFunc == SOFTMAX)
    {
        // find maximum value
        float max = 0.0f;
        for(int n = 0; n < net->layers[outputLayerIndex].neuronCount; n++)
        {
            if(net->layers[outputLayerIndex].neurons[n].activation > max)
            {
                max = net->layers[outputLayerIndex].neurons[n].activation;
            }
        }

        for(int n = 0; n < net->layers[outputLayerIndex].neuronCount; n++)
        {
            net->layers[outputLayerIndex].neurons[n].activation -= max;
        }

        float sum = 0.0f;
        for(int n = 0; n < net->layers[outputLayerIndex].neuronCount; n++)
        {
            sum += exp(net->layers[outputLayerIndex].neurons[n].activation);
        }

        for(int n = 0; n < net->layers[outputLayerIndex].neuronCount; n++)
        {
            net->layers[outputLayerIndex].neurons[n].activation = exp(net->layers[outputLayerIndex].neurons[n].activation) / sum;
        }
    }
}

enum CostFunctionType 
{
    MEAN_SQUARE_ERROR,
    CROSS_ENTROPY_LOSS
};

void BackPropagate(Net *net, float *target, unsigned int targetCount, enum CostFunctionType costFunctionType)
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
            net->layers[i].neurons[k].dcost_da = 0.0f;

            if (i == net->layerCount - 1)
            {
                if(costFunctionType == MEAN_SQUARE_ERROR)
                {
                    net->layers[i].neurons[k].dcost_da = 2 * (net->layers[net->layerCount - 1].neurons[k].activation - target[k]) / (float)targetCount;
                }
                else if(costFunctionType == CROSS_ENTROPY_LOSS)
                {
                    net->layers[i].neurons[k].dcost_da = net->layers[net->layerCount - 1].neurons[k].activation - target[k];
                }
            }
            else
            {
                for (int j = 0; j < net->layers[i + 1].neuronCount; j++)
                {
                    if(net->layers[i + 1].activationFunc == SOFTMAX)
                    {
                        float dcost_dz = net->layers[i + 1].neurons[j].dcost_da;
                        net->layers[i].neurons[k].dcost_da += net->layers[i + 1].neurons[j].weights[k].data * dcost_dz;
                    }
                    else
                    {
                        OneInputOneOuputFuncPtr derivative = GetActivationDerivativeFunction(net->layers[i + 1].activationFunc);
                        float da_dz = derivative(net->layers[i + 1].neurons[j].z);
                        float dcost_dz = da_dz * net->layers[i + 1].neurons[j].dcost_da;
                        net->layers[i].neurons[k].dcost_da += net->layers[i + 1].neurons[j].weights[k].data * dcost_dz;
                    }
                }
            }
        }
    }
}

void ComputeGradients(Net *net, int batchSize)
{
    for (int i = 1; i < net->layerCount; i++)
    {
        for (int j = 0; j < net->layers[i].neuronCount; j++)
        {
            float da_dz = 0.0f;
            if(net->layers[i].activationFunc == SOFTMAX)
            {
                da_dz = 1.0f;
            }
            else
            {
                OneInputOneOuputFuncPtr derivative = GetActivationDerivativeFunction(net->layers[i].activationFunc);
                da_dz = derivative(net->layers[i].neurons[j].z);
            }
            
            float dcost_dz = net->layers[i].neurons[j].dcost_da * da_dz;

            for (int k = 0; k < net->layers[i - 1].neuronCount; k++)
            {
                float gradient = (net->layers[i - 1].neurons[k].activation * dcost_dz) / (float)batchSize;
                net->layers[i].neurons[j].weights[k].grad += gradient;
            }

            net->layers[i].neurons[j].bias.grad += (dcost_dz / (float)batchSize);
        }
    }
}

void ZeroGradients(Net *net)
{
    for (int i = 1; i < net->layerCount; i++)
    {
        for (int j = 0; j < net->layers[i].neuronCount; j++)
        {
            for (int k = 0; k < net->layers[i - 1].neuronCount; k++)
            {
                net->layers[i].neurons[j].weights[k].grad = 0.0f;
            }

            net->layers[i].neurons[j].bias.grad = 0.0f;
        }
    }
}

void Update(Net *net)
{
    for (int i = 1; i < net->layerCount; i++)
    {
        for (int j = 0; j < net->layers[i].neuronCount; j++)
        {
            for (int k = 0; k < net->layers[i - 1].neuronCount; k++)
            {
                net->layers[i].neurons[j].weights[k].data -= net->learnRate * net->layers[i].neurons[j].weights[k].grad;
            }
            net->layers[i].neurons[j].bias.data -= net->learnRate * net->layers[i].neurons[j].bias.grad;
        }
    }
}

// mean squared error
float ComputeMSE(Net net, float *target, unsigned int targetCount)
{
    if (net.layers[net.layerCount - 1].neuronCount != targetCount)
    {
        printf("error: no.of neurons in output layer != target count\n");
        return 0;
    }

    float cost = 0;

    for (int i = 0; i < targetCount; i++)
    {
        float diff = target[i] - net.layers[net.layerCount - 1].neurons[i].activation; 
        cost += (diff * diff);
    }

    return cost / (float)targetCount;
}

float ComputeCrossEntropyLoss(Net net, float *target, unsigned int targetCount)
{
    if (net.layers[net.layerCount - 1].neuronCount != targetCount)
    {
        printf("error: no.of neurons in output layer != target count\n");
        return 0;
    }

    float sum = 0;

    for (int i = 0; i < targetCount; i++)
    {
        sum += target[i] * log(net.layers[net.layerCount - 1].neurons[i].activation + 0.000001f); 
    }

    return -sum;
}

void LogWeights(Net net)
{
    for (int i = 1; i < net.layerCount; i++)
    {
        for (int j = 0; j < net.layers[i].neuronCount; j++)
        {
            for (int k = 0; k < net.layers[i - 1].neuronCount; k++)
            {
                printf("layer: [%d] from: [%d] to: [%d] weight: %0.3f\n", i, k + 1, j + 1, net.layers[i].neurons[j].weights[k].data);
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

    fprintf(out, "%d\n", net.layerCount); // layer count

    // each layer info in the format: <neuron count> <activation function>
    for (int n = 0; n < net.layerCount; n++)
    {
        fprintf(out, "%d %s\n", net.layers[n].neuronCount,ActivationFunctionToString(net.layers[n].activationFunc));
    }

    for (int i = 1; i < net.layerCount; i++)
    {
        for (int j = 0; j < net.layers[i].neuronCount; j++)
        {
            for (int k = 0; k < net.layers[i - 1].neuronCount; k++)
            {
                fprintf(out, "%f\n", net.layers[i].neurons[j].weights[k].data);
            }
            fprintf(out, "%f\n", net.layers[i].neurons[j].bias.data);
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

    fscanf(in, "%d\n", &layerCount);

    for (int n = 0; n < layerCount; n++)
    {
        char functionName[20] = {0};
        int neuronCount = 0;
        fscanf(in, "%d %s\n", &neuronCount, functionName);
        if(n == 0)
        {
            AddLayer(&net, CreateLayer(0, neuronCount, LINEAR));
        } 
        else 
        {
            AddLayer(&net, CreateLayer(net.layers[net.layerCount - 1].neuronCount, neuronCount, StringToActivationFunction(functionName)));
        }
    }

    for (int i = 1; i < net.layerCount; i++)
    {
        for (int j = 0; j < net.layers[i].neuronCount; j++)
        {
            for (int k = 0; k < net.layers[i - 1].neuronCount; k++)
            {
                fscanf(in, "%f\n", &net.layers[i].neurons[j].weights[k].data);
            }
            fscanf(in, "%f\n", &net.layers[i].neurons[j].bias.data);
        }
    }

    fclose(in);

    return net;
}
