#pragma once

#include "Kernel.cuh"

void solve_base_kernel(uint8_t * query , uint8_t *  key , float *  LUT , float * store);


void solve_stream_kernel(uint8_t * query , uint8_t *  key , float *  LUT , float * store);

void FancyAllocation(uint8_t *query, uint8_t *key, float *LUT, float *store);

