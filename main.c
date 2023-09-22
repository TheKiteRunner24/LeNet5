#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "lenet.h"
#include "utils.h"


#define FILE_TRAIN_IMAGE		"train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL		"train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"
#define LENET_FILE 		"model.dat"
#define COUNT_TRAIN		60000
#define COUNT_TEST		10000


void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
	for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
	{
		TrainBatch(lenet, train_data + i, train_label + i, batch_size);
		if (i * 100 / total_size > percent)
			printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = i * 100 / total_size);
	}
}


int testing(LeNet5 *lenet, image *test_data, uint8 *test_label,int total_size)
{
	int right = 0, percent = 0;
	for (int i = 0; i < total_size; ++i)
	{
		uint8 l = test_label[i];
		int p = Predict(lenet, test_data[i], 10);
		right += l == p;
		if (i * 100 / total_size > percent)
			printf("test:%2d%%\n", percent = i * 100 / total_size);
	}
	return right;
}


void run()
{
	image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
	uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
	if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Folder Included the main.c\n");
		free(train_data);
		free(train_label);
	}
	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Folder Included the main.c\n");
		free(test_data);
		free(test_label);
	}

	LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
	if (load(lenet, LENET_FILE))
		Initial(lenet);
	clock_t start = clock();
	
    training(lenet, train_data, train_label, 300, COUNT_TRAIN);
	int right = testing(lenet, test_data, test_label, COUNT_TEST);

	printf("%d/%d\n", right, COUNT_TEST);
	printf("Time:%u\n", (unsigned)(clock() - start));
	save(lenet, LENET_FILE);
	free(lenet);
	free(train_data);
	free(train_label);
	free(test_data);
	free(test_label);
}


void test_only(){
	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Folder Included the main.c\n");
		free(test_data);
		free(test_label);
	}

	LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
	int ret = load(lenet, LENET_FILE);

	clock_t start = clock();
	int right = testing(lenet, test_data, test_label, COUNT_TEST);
	printf("%d/%d\n", right, COUNT_TEST);
	printf("Time:%u\n", (unsigned)(clock() - start));

	free(lenet);
	free(test_data);
	free(test_label);

}


int testing_quantized(LeNet5_quantized *lenet_q, image *test_data, uint8 *test_label,int total_size)
{
	int right = 0, percent = 0;
	for (int i = 0; i < total_size; ++i)
	{
		uint8 l = test_label[i];
		int p = Predict_quantized(lenet_q, test_data[i], 10);
		right += l == p;
		if (i * 100 / total_size > percent)
			printf("test:%2d%%\n", percent = i * 100 / total_size);
	}
	return right;
}


void test_only_quantized(){
	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Folder Included the main.c\n");
		free(test_data);
		free(test_label);
	}

	LeNet5_quantized *lenet_q = (LeNet5_quantized *)malloc(sizeof(LeNet5_quantized));
	int ret = load_quantized_model(lenet_q, "quantized_model.dat");

	clock_t start = clock();
	int right = testing_quantized(lenet_q, test_data, test_label, COUNT_TEST);
	printf("%d/%d\n", right, COUNT_TEST);
	printf("Time:%u\n", (unsigned)(clock() - start));

	free(lenet_q);
	free(test_data);
	free(test_label);
}



int main()
{
	//run(); // 训练和推理,生成高精度模型

	//test_only(); // 仅推理

	get_scale_and_quantize(); //由高精度模型生成低精度模型

	test_only_quantized(); // 使用低精度模型进行推理

	
	return 0;
}
