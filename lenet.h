﻿/*
@author : 范文捷
@data    : 2016-04-20
@note	: 根据Yann Lecun的论文《Gradient-based Learning Applied To Document Recognition》编写
@api	:

批量训练
void TrainBatch(LeNet5 *lenet, image *inputs, const char(*resMat)[OUTPUT],uint8 *labels, int batchSize);

训练
void Train(LeNet5 *lenet, image input, const char(*resMat)[OUTPUT],uint8 label);

预测
uint8 Predict(LeNet5 *lenet, image input, const char(*resMat)[OUTPUT], uint8 count);

初始化
void Initial(LeNet5 *lenet);
*/

#pragma once

#define LENGTH_KERNEL	5

#define LENGTH_FEATURE0	32
#define LENGTH_FEATURE1	(LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2	(LENGTH_FEATURE1 >> 1)
#define LENGTH_FEATURE3	(LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
#define	LENGTH_FEATURE4	(LENGTH_FEATURE3 >> 1)
#define LENGTH_FEATURE5	(LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

#define INPUT			1
#define LAYER1			6
#define LAYER2			6
#define LAYER3			16
#define LAYER4			16
#define LAYER5			120
#define OUTPUT          10

#define ALPHA 0.5
#define PADDING 2
//#define PADDING 0

typedef unsigned char uint8;
typedef signed char int8;
typedef signed int int32;

typedef uint8 image[28][28];

typedef struct LeNet5
{
	double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];

	double bias0_1[LAYER1];
	double bias2_3[LAYER3];
	double bias4_5[LAYER5];
	double bias5_6[OUTPUT];

}LeNet5;

typedef struct Feature
{
	double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
	double layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
	double layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
	double layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
	double layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
	double layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
	double output[OUTPUT];
}Feature;

typedef struct LeNet5_quantized
{
	int8 weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
	int8 weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
	int8 weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
	int8 weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];

	// 偏置涉及的计算量较小，先不量化
	double bias0_1[LAYER1];
	double bias2_3[LAYER3];
	double bias4_5[LAYER5];
	double bias5_6[OUTPUT];

	double c1_scale;
	double c2_scale;
	double c3_scale;
	double fc_scale;

}LeNet5_quantized;

typedef struct Feature_quantized
{
	double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
	int8 input_q[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
	//卷积
	int32 layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
	double layer1_f[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
	int8 layer1_q[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
	//池化
	int8 layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
	//卷积
	int32 layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
	double layer3_f[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
	int8 layer3_q[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
	//池化
	int8 layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
	//卷积
	int32 layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
	double layer5_f[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
	int8 layer5_q[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
	//全连接
	int32 output[OUTPUT];
	double output_f[OUTPUT];

}Feature_quantized;


void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize);

//void Train(LeNet5 *lenet, image input, uint8 label);

uint8 Predict(LeNet5 *lenet, image input, uint8 count);

uint8 Predict_quantized(LeNet5_quantized *lenet_q, image input, uint8 count);

void Initial(LeNet5 *lenet);
