#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "lenet.h"

#pragma once


int load(LeNet5 *lenet, char filename[]);
int save(LeNet5 *lenet, char filename[]);
int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[]);

int load_quantized_model(LeNet5_quantized *lenet_q, char filename[]);
int save_quantized_model(LeNet5_quantized *lenet_q, char filename[]);
void get_scale_and_quantize();
void print_quantized_model(LeNet5_quantized *lenet_q, char filename[]);