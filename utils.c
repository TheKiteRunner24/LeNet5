#include "utils.h"

#define LENET_FILE 		"model.dat"


// 得到每个权重矩阵的scale并量化到int8，保存量化后的weights以及scale
void get_scale_and_quantize(){
	LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
	FILE *fp = fopen(LENET_FILE, "rb");
	if (fp == NULL) {
        printf("无法打开文件\n");
        return;
    }

	fread(lenet, sizeof(LeNet5), 1, fp);

	LeNet5_quantized *lenet_q = (LeNet5_quantized *)malloc(sizeof(LeNet5_quantized));

	double c1_min = lenet->weight0_1[0][0][0][0];
    double c1_max = lenet->weight0_1[0][0][0][0];
	double c2_min = lenet->weight2_3[0][0][0][0];
    double c2_max = lenet->weight2_3[0][0][0][0];
	double c3_min = lenet->weight4_5[0][0][0][0];
    double c3_max = lenet->weight4_5[0][0][0][0];
	double fc_min = lenet->weight5_6[0][0];
    double fc_max = lenet->weight5_6[0][0];

	for (int i = 0; i < INPUT; i++) {
        for (int j = 0; j < LAYER1; j++) {
            for (int k = 0; k < LENGTH_KERNEL; k++) {
                for (int l = 0; l < LENGTH_KERNEL; l++) {
                    double value = lenet->weight0_1[i][j][k][l];
                    if (value < c1_min) {
                        c1_min = value;
                    }
                    if (value > c1_max) {
                        c1_max = value;
                    }
                }
            }
        }
    }

	// 对称量化
	double c1_absmax = fabs(c1_max) > fabs(c1_min) ? fabs(c1_max) : fabs(c1_min);
    lenet_q->c1_scale = c1_absmax / 127;

    // q = r/S
	for (int i = 0; i < INPUT; i++) {
        for (int j = 0; j < LAYER1; j++) {
            for (int k = 0; k < LENGTH_KERNEL; k++) {
                for (int l = 0; l < LENGTH_KERNEL; l++) {
					lenet_q->weight0_1[i][j][k][l] = (int8)(lenet->weight0_1[i][j][k][l] / lenet_q->c1_scale);
                }
            }
        }
    }

	//=========================================================================

	for (int i = 0; i < LAYER2; i++) {
        for (int j = 0; j < LAYER3; j++) {
            for (int k = 0; k < LENGTH_KERNEL; k++) {
                for (int l = 0; l < LENGTH_KERNEL; l++) {
                    double value = lenet->weight2_3[i][j][k][l];
                    if (value < c2_min) {
                        c2_min = value;
                    }
                    if (value > c2_max) {
                        c2_max = value;
                    }
                }
            }
        }
    }

	double c2_absmax = fabs(c2_max) > fabs(c2_min) ? fabs(c2_max) : fabs(c2_min);
    lenet_q->c2_scale = c2_absmax / 127;

	for (int i = 0; i < LAYER2; i++) {
        for (int j = 0; j < LAYER3; j++) {
            for (int k = 0; k < LENGTH_KERNEL; k++) {
                for (int l = 0; l < LENGTH_KERNEL; l++) {
                    lenet_q->weight2_3[i][j][k][l] = (int8)(lenet->weight2_3[i][j][k][l] / lenet_q->c2_scale);
                }
            }
        }
    }

	//=================================================================

	for (int i = 0; i < LAYER4; i++) {
        for (int j = 0; j < LAYER5; j++) {
            for (int k = 0; k < LENGTH_KERNEL; k++) {
                for (int l = 0; l < LENGTH_KERNEL; l++) {
                    double value = lenet->weight4_5[i][j][k][l];
                    if (value < c3_min) {
                        c3_min = value;
                    }
                    if (value > c3_max) {
                        c3_max = value;
                    }
                }
            }
        }
    }

	double c3_absmax = fabs(c3_max) > fabs(c3_min) ? fabs(c3_max) : fabs(c3_min);
    lenet_q->c3_scale = c3_absmax / 127;

	for (int i = 0; i < LAYER4; i++) {
        for (int j = 0; j < LAYER5; j++) {
            for (int k = 0; k < LENGTH_KERNEL; k++) {
                for (int l = 0; l < LENGTH_KERNEL; l++) {
                    lenet_q->weight4_5[i][j][k][l] = (int8)(lenet->weight4_5[i][j][k][l] / lenet_q->c3_scale);
                }
            }
        }
    }

	//=====================================================================

	for (int i = 0; i < LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5; i++) {
        for (int j = 0; j < OUTPUT; j++) {
			double value = lenet->weight5_6[i][j];
			if (value < fc_min) {
				fc_min = value;
			}
			if (value > fc_max) {
				fc_max = value;
			}
        }
    }

	double fc_absmax = fabs(fc_max) > fabs(fc_min) ? fabs(fc_max) : fabs(fc_min);
    lenet_q->fc_scale = fc_absmax / 127;

	for (int i = 0; i < LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5; i++) {
        for (int j = 0; j < OUTPUT; j++) {
			lenet_q->weight5_6[i][j] = (int8)(lenet->weight5_6[i][j] / lenet_q->fc_scale);
        }
    }

	save_quantized_model(lenet_q, "quantized_model.dat");

	free(lenet);
	free(lenet_q);
	fclose(fp);
}


int save_quantized_model(LeNet5_quantized *lenet_q, char filename[])
{
	FILE *fp = fopen(filename, "wb");
	if (!fp) return 1;
	fwrite(lenet_q, sizeof(LeNet5_quantized), 1, fp);
	fclose(fp);
	return 0;
}


int load_quantized_model(LeNet5_quantized *lenet_q, char filename[])
{
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	fread(lenet_q, sizeof(LeNet5_quantized), 1, fp);
	fclose(fp);
	return 0;
}


void print_quantized_model(LeNet5_quantized *lenet_q, char filename[]){

	FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("无法打开文件\n");
        return;
    }

	fprintf(file, "weight0_1:\n");
    for (int i = 0; i < INPUT; i++) {
        for (int j = 0; j < LAYER1; j++) {
            for (int k = 0; k < LENGTH_KERNEL; k++) {
                for (int l = 0; l < LENGTH_KERNEL; l++) {
                    fprintf(file, "%d ", lenet_q->weight0_1[i][j][k][l]);
                }
            }
            fprintf(file, "\n");
        }
		fprintf(file, "\n");
    }

	fprintf(file, "weight2_3:\n");
    for (int i = 0; i < LAYER2; i++) {
        for (int j = 0; j < LAYER3; j++) {
            for (int k = 0; k < LENGTH_KERNEL; k++) {
                for (int l = 0; l < LENGTH_KERNEL; l++) {
                    fprintf(file, "%d ", lenet_q->weight2_3[i][j][k][l]);
                }
            }
            fprintf(file, "\n");
        }
		fprintf(file, "\n");
    }

	fprintf(file, "weight4_5:\n");
    for (int i = 0; i < LAYER4; i++) {
        for (int j = 0; j < LAYER5; j++) {
            for (int k = 0; k < LENGTH_KERNEL; k++) {
                for (int l = 0; l < LENGTH_KERNEL; l++) {
                    fprintf(file, "%d ", lenet_q->weight4_5[i][j][k][l]);
                }  
            }
            fprintf(file, "\n");
        }
		fprintf(file, "\n");
    }

	fprintf(file, "\n");
	fprintf(file, "weight5_6:\n");
    for (int i = 0; i < LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5; i++) {
        for (int j = 0; j < OUTPUT; j++) { 
            fprintf(file, "%d ", lenet_q->weight5_6[i][j]); 
        }
		fprintf(file, "\n");
    }

	fprintf(file, "\n");
	fprintf(file, "bias0_1:\n");
	for (int i = 0; i < LAYER1; i++) { 
		fprintf(file, "%lf ", lenet_q->bias0_1[i]); 
	}
	fprintf(file, "\n");

	fprintf(file, "\n");
	fprintf(file, "bias2_3:\n");
	for (int i = 0; i < LAYER3; i++) { 
		fprintf(file, "%lf ", lenet_q->bias2_3[i]); 
	}
	fprintf(file, "\n");

	fprintf(file, "\n");
	fprintf(file, "bias4_5:\n");
	for (int i = 0; i < LAYER5; i++) { 
		fprintf(file, "%lf ", lenet_q->bias4_5[i]); 
	}
	fprintf(file, "\n");

	fprintf(file, "\n");
	fprintf(file, "bias5_6:\n");
	for (int i = 0; i < OUTPUT; i++) { 
		fprintf(file, "%lf ", lenet_q->bias5_6[i]); 
	}
	fprintf(file, "\n");

	fclose(file);
}


int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(*data)*count, 1, fp_image);
	fread(label,count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}

int save(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "wb");
	if (!fp) return 1;
	fwrite(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}


int load(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	fread(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}