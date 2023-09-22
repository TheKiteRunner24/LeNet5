#include <stdlib.h>
#include "lenet.h"
#include "cnnapi_base_q.h"

#pragma once

void lenet_forward(LeNet5_quantized *lenet, Feature_quantized *features);