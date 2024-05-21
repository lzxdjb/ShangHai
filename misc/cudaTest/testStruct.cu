#include <stdio.h>




struct bs {
    int* ptr;
    bs(int* pt = nullptr) : ptr(pt) {}
};




// Note: kernel can only be called once source-output computation has been handled.
// There is no functionality for source-handling from this kernel.
__global__ void kernel(int* arr) {
    //blockIdx.x * blockDim.x + threadIdx.x;
    //printf("\n\n|GPUGPUGPUGPUGPUGPU|\n");
    printf("\n\nOn thread %d, arr contains %d. blockDim is: %d, blockIdx is: %d", threadIdx.x, arr[threadIdx.x], blockDim.x, blockIdx.x); 
}

__global__ void structKrnl(bs strct) {
    printf("\n\nOn thread %d, arr contains %d. blockDim is: %d, blockIdx is: %d", threadIdx.x, strct.ptr[threadIdx.x], blockDim.x, blockIdx.x);
}



#define N   5
int main() {

    //int kek[N]{11,21,31,41,51};
    //int* k;

    int* jej = new int[N]{13,23,33,43,53};
    int* j;
    bs test1 = bs(jej);

    /*cudaMalloc((void**)&k, N*sizeof(int));
    cudaMemcpy(k, jej, N*sizeof(int), cudaMemcpyHostToDevice); 
    kernel<<<1,N>>>(k);
    cudaDeviceSynchronize();  
    cudaFree(k);*/

    cudaMalloc((void**)&j, sizeof(int));
    cudaMemcpy(j, jej, sizeof(int), cudaMemcpyHostToDevice);
    test1.ptr = j;
    structKrnl<<<1,N>>>(test1);
    cudaMemcpy(jej, j, sizeof(int), cudaMemcpyDeviceToHost);
    test1.ptr = jej;
    cudaFree(j);

    return 0;
}






