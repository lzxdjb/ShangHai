#include <stdio.h>

struct StructA {
    int** arr;
};

#define N 10

// pass struct by value (may not be efficient for complex structures)
__global__ void kernel2(StructA in) {
    in.arr[blockIdx.x][threadIdx.x] *= 2;
    printf("at block #%d, thread #%d, with vals: %d\n", blockIdx.x, threadIdx.x, in.arr[blockIdx.x][threadIdx.x]);
}


int main() {
    int h_1[N] = {1,2,3,4,5,6,7,8,9,10};
    int h_2[N] = {11,22,33,44,55,66,77,88,99,100};
    int* h_x[2] = {h_1,h_2};

    StructA h_a;
    int** d_x;
    int* d_1;
    int* d_2;

    // 1. Allocate device array.
    cudaMalloc((void**) &(d_x), sizeof(int*)*2);
    cudaMalloc((void**) &(d_1), sizeof(int)*N);
    cudaMalloc((void**) &(d_2), sizeof(int)*N);

    // 2. Copy array contents from host to device.
    cudaMemcpy(d_1, h_1, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_2, h_2, sizeof(int)*N, cudaMemcpyHostToDevice);
    h_x[0] = d_1;
    h_x[1] = d_2;
    
    cudaMemcpy(d_x, h_x, sizeof(int*)*2, cudaMemcpyHostToDevice);


    // 3. Point to device pointer in host struct.
    h_a.arr = d_x;

    // 4. Call kernel with host struct as argument
    kernel2<<<2, N>>>(h_a);

    cudaDeviceSynchronize();

    // 5. Copy pointer from device to host.
    cudaMemcpy(h_1, d_1, sizeof(int)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_2, d_2, sizeof(int)*N, cudaMemcpyDeviceToHost);

    // 6. Point to host pointer in host struct
    //    (or do something else with it if this is not needed)
    h_x[0] = h_1;
    h_x[1] = h_2;
    h_a.arr = h_x;

    for (int i=0; i<10; i++) {
        for (int j=0; j<2; j++) {
            printf("\n %d", h_a.arr[j][i]);
        }
    }

}





