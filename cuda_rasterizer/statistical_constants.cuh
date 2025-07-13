#pragma once

#define MAX_STD_SAMPLES 1000

#ifndef STATISTICAL_CONSTANTS_IMPL
extern __constant__ float BASE_SAMPLES_MAX[MAX_STD_SAMPLES * 3];
extern __constant__ int   NUM_STD_SAMPLES;
#endif