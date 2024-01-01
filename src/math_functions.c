#include <math.h>

float linear(float z)
{
    return z;
}

float dlinear_dz(float z)
{
    return 1.0f;
}

// RELU
float ReLU(float z)
{
    return z > 0 ? z : 0;
}

float dReLU_dz(float z)
{
    return z > 0 ? 1 : 0;
}

// sigmoid
float sigmoid(float z)
{
    return exp(z) / (1 + exp(z));
}

float dsigmoid_dz(float z)
{
    return sigmoid(z) * (1 - sigmoid(z));
}

// tanh
float dtanh_dz(float z)
{
    float x = tanhf(z);
    return 1 - (x * x);
}
