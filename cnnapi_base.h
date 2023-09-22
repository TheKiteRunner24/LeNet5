#include <stdlib.h>

#pragma once

void convolution_valid_base(const double* input, int input_size,
                        const double* kernel, int kernel_size,
                        double* output);

void fully_connected_base(const double* input, const double* weights, 
                    int num_left, int num_right, 
                    double* output);

