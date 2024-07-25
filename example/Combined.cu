#include "src/FancyDesign.cuh"
#include "src/consolidate.cuh"
#include "src/GancyDesign.cuh"
#include <unistd.h>

using namespace std;

void DebugQ(uint8_t *query)
{
    for (int b = 0; b < bs; b++)
    {
        for (int n = 0; n < nh; n++)
        {
            std::cout << endl;
            for (int q = 0; q < qlen; q++)
            {
                for (int d = 0; d < dim; d++)
                {
                    std::cout << static_cast<int>(query[b * nh * qlen * dim + n * qlen * dim + q * dim + d]) << " ";
                    // std::cout<<static_cast<int>(uint8_t(1));
                }
                std::cout << endl;
            }
        }
    }
}

void DebugK(uint8_t *query)
{
    for (int b = 0; b < bs; b++)
    {
        for (int n = 0; n < nh; n++)
        {
            std::cout << endl;
            for (int k = 0; k < klen; k++)
            {
                for (int d = 0; d < dim; d++)
                {
                    std::cout << static_cast<int>(query[b * nh * klen * dim + n * klen * dim + k * dim + d]) << " ";
                    // std::cout<<static_cast<int>(uint8_t(1));
                }
                std::cout << endl;
            }
        }
    }
}

void DebugLUT(float *LUT)
{
    for (int n = 0; n < nh; n++)
    {
        std::cout << endl;
        for (int k1 = 0; k1 < K; k1++)
        {
            for (int k2 = 0; k2 < K; k2++)
            {
                std::cout << static_cast<int>(LUT[n * K * K + k1 * K + k2]) << " ";
            }
            std::cout << endl;
        }
    }
}

void DebugStore(float *store)
{
    for (int b = 0; b < bs; b++)
    {
        for (int n = 0; n < nh; n++)
        {
            std::cout << endl;
            for (int q = 0; q < qlen; q++)
            {
                for (int k = 0; k < klen; k++)
                {
                    std::cout << static_cast<int>(store[b * nh * qlen * klen + n * qlen * klen + q * klen + k]) << " ";
                }
            }
        }
    }
}

void Initial(uint8_t *query, uint8_t *key, float *LUT, float *store)
{
    for (int b = 0; b < bs; b++)
    {
        for (int n = 0; n < nh; n++)
        {
            for (int q = 0; q < qlen; q++)
            {
                for (int d = 0; d < dim; d++)
                {
                    query[b * nh * qlen * dim + n * qlen * dim + q * dim + d] = 1;
                }
            }
        }
    }

    for (int b = 0; b < bs; b++)
    {
        for (int n = 0; n < nh; n++)
        {
            for (int k = 0; k < klen; k++)
            {
                for (int d = 0; d < dim; d++)
                {
                    key[b * nh * klen * dim + n * klen * dim + k * dim + d] = 2;
                }
            }
        }
    }

    for (int n = 0; n < nh; n++)
    {
        for (int k1 = 0; k1 < K; k1++)
        {
            for (int k2 = 0; k2 < K; k2++)
            {
                LUT[n * K * K + k1 * K + k2] = 3;
            }
        }
    }

    //////////// No need to do
    for (int b = 0; b < bs; b++)
    {
        for (int n = 0; n < nh; n++)
        {
            for (int q = 0; q < qlen; q++)
            {
                for (int k = 0; k < klen; k++)
                {
                    store[b * nh * qlen * klen + n * qlen * klen + q * klen + k] = 10;
                }
            }
        }
    }
}

int main()
{

    uint8_t *query = (uint8_t *)malloc(sizeof(uint8_t) * bs * nh * qlen * dim);

    uint8_t *key = (uint8_t *)malloc(sizeof(uint8_t) * bs * nh * klen * dim);

    float *LUT = (float *)malloc(sizeof(float) * nh * K * K);

    float *store;
    checkCudaErrors(cudaMallocHost((void **)&store, sizeof(float) * bs * nh * qlen * klen));

    Initial(query, key, LUT, store);

    for (int i = 0; i < 3; i++)
    {

        solve_base_kernel(query, key, LUT, store);
        cudaDeviceSynchronize();

        solve_stream_kernel(query, key, LUT, store);
        cudaDeviceSynchronize();

        MoreFancy(query, key, LUT, store);
        cudaDeviceSynchronize();

        FancyStream(query , key , LUT , store);
        cudaDeviceSynchronize();

    }


}