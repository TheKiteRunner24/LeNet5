#include "cnnapi_cvp.h"

int divideAndRoundUp(int A, int B) {
    if(A % B == 0)
        return A / B;
    else
        return A / B + 1;
}



void convolution_valid(const double* input, int input_size,
                        const double* kernel, int kernel_size,
                        double* output){




    int num_elem_per_reg = 8; // 64/8

    int num_reg = divideAndRoundUp(input_size, num_elem_per_reg);

    int calculate_times_per_kernel = divideAndRoundUp(kernel_size, num_elem_per_reg);

    for (int i = 0; i < input_size/25; i++)
    {
        output[i] = 0;
    }
    
    for (int i = 0; i < input_size; ) {
        for (int j = 0; j < kernel_size; j++) {
            output[i/kernel_size] += input[i] * kernel[j];
            i++;
        }
    }

    

 

}

