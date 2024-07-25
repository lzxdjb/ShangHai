#pragma once

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

typedef double tinytype;  

const int bs = 32;
const int nh = 32;  

// bs * nh must be bigger than stream size.

const int qlen = 256; 
const int klen = 256;
const int dim = 5;  

// const int bits = 4;  
// const int K = 2 << bits;

const int bits = 4;  
const int K = 64;

const int block = bs * nh;

const int nStreams = nh;
const int StreamModulnStreams = 4;



const int Fancyworkload = bs; // workload should be the factor of bs
const int FancyStreamWorkLoad = (bs * nh) / StreamModulnStreams; //You can change;


struct DebugStruct{
    float LUT[K * K];

    u_int8_t query[dim];
    u_int8_t key[dim];

    u_int8_t EncodingK;
    u_int8_t EncodingQ;
};

struct StreamStruct{
    float LUT[K * K];
    uint8_t query[bs * qlen * dim];
    uint8_t key[bs * klen * dim];
    float Store[bs * qlen * klen];

};












