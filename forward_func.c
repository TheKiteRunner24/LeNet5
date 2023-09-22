#include "forward_func.h"


// forward_func与lenet_forward的行为保持一致，训练得到模型使用forward_dunc

double ReLU_base(double x)
{
	return x*(x > 0);
}

void prepare_conv_data_base(const double* input, int inputRows, int inputCols,
                         int kernelRows, int kernelCols,
                         double* input_prepared){

    int outputRows = inputRows - kernelRows + 1;
    int outputCols = inputCols - kernelCols + 1;

    int idx = 0;
    for (int i = 0; i < outputRows; i++) {
        for (int j = 0; j < outputCols; j++) {
            for (int m = 0; m < kernelRows; m++) {
                for (int n = 0; n < kernelCols; n++) {
                    input_prepared[idx++] = input[(i + m) * inputCols + (j + n)]; 
                }
            }
        }
    }
}

void forward_func(LeNet5 *lenet, Feature *features)
{
	// 卷积层1 ============================================================================================
	
	for (int x = 0; x < (sizeof(lenet->weight0_1) / sizeof(*(lenet->weight0_1))); ++x) {  //1个图像
		for (int y = 0; y < (sizeof(*lenet->weight0_1) / sizeof(*(*lenet->weight0_1))); ++y) {  //6个卷积核

			double* input = (double*)features->input[x];
			double* kernel = (double*)lenet->weight0_1[x][y];
			double* output = (double*)features->layer1[y];
			
			//输入图像加了padding变成32*32
			int kernel_size = 25;
			int input_size = (32-5+1) * (32-5+1) * 25;

			double* input_prepared= (double*)calloc(input_size, sizeof(double));

			prepare_conv_data_base(input, 32, 32, 5, 5, input_prepared);
			convolution_valid_base(input_prepared, input_size, kernel, kernel_size, output);

			free(input_prepared);
		}
	}

	
	for (int j = 0; j < (sizeof(features->layer1) / sizeof(*(features->layer1))); ++j) { //6
		for (int i = 0; i < (sizeof(features->layer1[j]) / sizeof(double)); ++i) { //784
			((double *)features->layer1[j])[i] = ReLU_base(((double *)features->layer1[j])[i] + lenet->bias0_1[j]);
		}
	}

	// 池化层2 ============================================================================================

	int pool_size = 2;

    for (int i = 0; i < sizeof(features->layer2) / sizeof(features->layer2[0]); ++i) { //6
        for (int o0 = 0; o0 < sizeof(features->layer2[0]) / sizeof(features->layer2[0][0]); ++o0) {  //14
            for (int o1 = 0; o1 < sizeof(features->layer2[0][0]) / sizeof(features->layer2[0][0][0]); ++o1) { //14
                int inputRow = o0 * 2;
                int inputCol = o1 * 2;
                double maxVal = features->layer1[i][inputRow][inputCol];
                for (int m = 0; m < pool_size; m++) {
                    for (int n = 0; n < pool_size; n++) {
                        double currentVal = features->layer1[i][inputRow + m][inputCol + n];
                        if (currentVal > maxVal) {
                            maxVal = currentVal;
                        }
                    }
                }
                features->layer2[i][o0][o1] = maxVal;
            }
        }
    }

    // 卷积层3 ============================================================================================

    // 输入图像 6 * 14 * 14
    for (int x = 0; x < 6; ++x) {  //6个图像
		for (int y = 0; y < 16; ++y) {  //16个卷积核

			double* input = (double*)features->layer2[x];
			double* kernel = (double*)lenet->weight2_3[x][y];
			double* output = (double*)features->layer3[y];
			
			int kernel_size = 25;
			int input_size = (14-5+1) * (14-5+1) * 25;

			double* input_prepared= (double*)calloc(input_size, sizeof(double));

			prepare_conv_data_base(input, 14, 14, 5, 5, input_prepared);
			convolution_valid_base(input_prepared, input_size, kernel, kernel_size, output);

			free(input_prepared);
		}
	}

    for (int j = 0; j < (sizeof(features->layer3) / sizeof(*(features->layer3))); ++j) {
        for (int i = 0; i < (sizeof(features->layer3[j]) / sizeof(double)); ++i) {
            ((double *)features->layer3[j])[i] = ReLU_base(((double *)features->layer3[j])[i] + lenet->bias2_3[j]);
        }
    }



    // 池化层4 ============================================================================================

    for (int i = 0; i < 16; ++i) {
        for (int o0 = 0; o0 < 5; ++o0) {
            for (int o1 = 0; o1 < 5; ++o1) {
                int inputRow = o0 * 2;
                int inputCol = o1 * 2;
                double maxVal = features->layer3[i][inputRow][inputCol];
                for (int m = 0; m < pool_size; m++) {
                    for (int n = 0; n < pool_size; n++) {
                        double currentVal = features->layer3[i][inputRow + m][inputCol + n];
                        if (currentVal > maxVal) {
                            maxVal = currentVal;
                        }
                    }
                }
                features->layer4[i][o0][o1] = maxVal;
            }
        }
    }
    
    
    // 卷积层5 =====================================================================================================
    
    for (int x = 0; x < 16; ++x) {
        for (int y = 0; y < 120; ++y) {
            double* input = (double*)features->layer4[x];
			double* kernel = (double*)lenet->weight4_5[x][y];
			double* output = (double*)features->layer5[y];

			int kernel_size = 25;
			int input_size = 25;

            convolution_valid_base(input, input_size, kernel, kernel_size, output);
        }
    }


    for (int j = 0; j < (sizeof(features->layer5) / sizeof(*(features->layer5))); ++j) {
        for (int i = 0; i < (sizeof(features->layer5[j]) / sizeof(double)); ++i) {
            ((double *)features->layer5[j])[i] = ReLU_base(((double *)features->layer5[j])[i] + lenet->bias4_5[j]);
        }
    }

    
    // 全连接层 =====================================================================================================================================
    

    double* input = (double*)features->layer5;
    double* weights = (double*)lenet->weight5_6;
    double* output = (double*)features->output;

    fully_connected_base(input, weights, 120, 10, output);


    for (int j = 0; j < (sizeof(lenet->bias5_6) / sizeof(*(lenet->bias5_6))); ++j) { //10
        ((double *)features->output)[j] = ReLU_base(((double *)features->output)[j] + lenet->bias5_6[j]);
    }

}




