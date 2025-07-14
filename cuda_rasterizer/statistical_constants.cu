#define STATISTICAL_CONSTANTS_IMPL
#include "statistical_constants.cuh"

__device__ float BASE_SAMPLES_MAX[MAX_STD_SAMPLES * 3] = {0};
__device__ int   NUM_STD_SAMPLES                  = 0;
