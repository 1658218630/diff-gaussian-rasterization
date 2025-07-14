#pragma once

#define MAX_STD_SAMPLES 1000

#ifndef STATISTICAL_CONSTANTS_IMPL
extern __device__ float BASE_SAMPLES_MAX[MAX_STD_SAMPLES * 3];
extern __device__ int   NUM_STD_SAMPLES;
#endif