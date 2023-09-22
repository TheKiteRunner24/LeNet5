#include "cnnapi_base.h"

void convolution_valid_base(const double* input, int input_size,
                        const double* kernel, int kernel_size,
                        double* output){

    int i = 0;
    while (i < input_size) {
        for (int j = 0; j < kernel_size; j++, i++) {
            output[i / kernel_size] += input[i] * kernel[j];
        }
    }
}

void fully_connected_base(const double* input, const double* weights, 
                    int num_left, int num_right, 
                    double* output){

    for (int i = 0; i < num_left * num_right; i++){
            output[i % num_right] += input[i / num_right] * weights[i];
    }
}


