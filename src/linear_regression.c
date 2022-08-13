#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float testdata[4][3] = {
    {0,0,0},
    {0,1,1},
    {1,0,1},
    {1,1,1}
};

float sigmoid(float z) 
{
	return exp(z) / (1 + exp(z));
}

float dsigmoid_dz(float z) 
{
	return sigmoid(z) * (1 - sigmoid(z));
}

int main(int argc, char* argv[]) 
{
    float z = 0.0;
    float y = 0.0;
    
    float dcost_dw1 = 0.0;
    float dcost_dw2 = 0.0;
    float dcost_db = 0.0;
    float dcost_dy = 0.0;
    float dy_dz = 0.0;
    float dz_dw1 = 0.0;
    float dz_dw2 = 0.0;
    float dz_db = 0.0;
    
    float cost = 0.0;
    float learn_rate = 1.0;
    
	float w1 = rand() % 5;
	float w2 = rand() % 5;
	float bias = rand() % 5;
    
	for(int iteration = 0; iteration < 50000; iteration++) 
    {
        int i = 0;
        for(i = 0; i < 4; i++)
        {
            z = w1 * testdata[i][0] + w2 * testdata[i][1] + bias;
            y = sigmoid(z);
            
            cost = (y - testdata[i][2]) * (y - testdata[i][2]) / 2.0;
            dcost_dy = y - testdata[i][2];
            
            dy_dz = dsigmoid_dz(z);
            dz_dw1 = testdata[i][0];
            dz_dw2 = testdata[i][1];
            dz_db = 1;
            
            dcost_dw1 = dcost_dy * dy_dz * dz_dw1;
            dcost_dw2 = dcost_dy * dy_dz * dz_dw2;
            dcost_db = dcost_dy * dy_dz * dz_db;
            
            w1 = w1 - learn_rate * dcost_dw1;
            w2 = w2 - learn_rate * dcost_dw2;
            bias = bias - learn_rate * dcost_db;
            
            if (iteration % 10000 == 0) 
            {
                printf("iteration : %d ", iteration);
                printf("[w1 : %lf] [w2 : %lf] [bias : %lf] [x1 : %lf, x2 : %lf, output : %lf] cost : %lf\n", w1, w2, bias, testdata[i][0], testdata[i][1], y, cost);
            }
        }
    }
        
    return 0;
}
