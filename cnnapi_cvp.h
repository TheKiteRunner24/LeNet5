#include "lenet.h"

void convolution_valid(const double* input, int input_size,
                        const double* kernel, int kernel_size,
                        double* output);