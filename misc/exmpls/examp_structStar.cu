#include <stdio.h>

struct StructA {
    int* arr;
};

#define N 10

// pass struct by value (may not be efficient for complex structures)
__global__ void kernel2(StructA in) {
    in.arr[threadIdx.x] *= 2;
    printf("at thread #%d with entry %d\n", threadIdx.x, in.arr[threadIdx.x]);
}


int main() {
    int h_1[N] = {1,2,3,4,5,6,7,8,9,10};

    StructA h_a;
    int* d_arr;

    // 1. Allocate device array.
    cudaMalloc((void**) &(d_arr), sizeof(int)*N);

    // 2. Copy array contents from host to device.
    cudaMemcpy(d_arr, h_1, sizeof(int)*N, cudaMemcpyHostToDevice);

    // 3. Point to device pointer in host struct.
    h_a.arr = d_arr;

    // 4. Call kernel with host struct as argument
    kernel2<<<1, N>>>(h_a);

    cudaDeviceSynchronize();

    // 5. Copy pointer from device to host.
    cudaMemcpy(h_1, d_arr, sizeof(int)*N, cudaMemcpyDeviceToHost);

    // 6. Point to host pointer in host struct
    //    (or do something else with it if this is not needed)
    h_a.arr = h_1;

    for (int i=0; i<10; i++) {
        printf("\n %d", h_a.arr[i]);
    }
}





