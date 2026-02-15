#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

__global__ void HistogramKernel(uint32_t *result_array, uint32_t *input_array, size_t inputWidth)
{
    const int numThreadsTotal = gridDim.x * gridDim.y * blockDim.x * blockDim.y; // threads total
    const int numThreadsBlock = blockDim.x * blockDim.y;
    const int uniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;
    const int threadId = threadIdx.y * blockDim.x + threadIdx.x; // flattened within a block
    const int globalTid = uniqueBlockIndex * numThreadsBlock + threadId; // flattened in whole invocation

    __shared__ uint32_t blockSharedHist[HISTO_HEIGHT * HISTO_WIDTH];

    // if(globalTid == 1){
    //     printf("numThreadsTotal %d\n", numThreadsTotal);
    //     printf("GPU input array:\n");
    //     for(int i = 0; i < INPUT_HEIGHT; i++){
    //         for(int j = 0; j < INPUT_WIDTH; j++){
    //             printf("%d ", input_array[i*inputWidth+j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n\n");
    // }
    // init shared histogram for this block
    // TODO: unroll?
    for(int pos = threadId; pos < HISTO_HEIGHT * HISTO_WIDTH; pos += numThreadsBlock){
        blockSharedHist[pos] = 0;
    }

    __syncthreads();

    // read from input and update block histogram
    for(int pos = globalTid; pos < INPUT_HEIGHT * inputWidth; pos += numThreadsTotal){ // should have enough threads
        int col = pos % inputWidth;
        if(col < INPUT_WIDTH) {// overflow threads on edges
            uint32_t data = input_array[pos];

            if(blockSharedHist[data] < 255){ // try to limit unecessary atomicAdds
                uint32_t test = atomicAdd(blockSharedHist + data, 1);
                if(test >= 255){ // must test to not over add
                    blockSharedHist[data] = 255;
                }
            }
        }
    }
    __syncthreads();

    // if(globalTid == 1){
    //     printf("print block histogram: \n");
    //     for(int i = 0; i < 1024; i++)
    //         printf("%d ", blockSharedHist[i]);
    //     printf("\n\n");

    // }

    for(int pos = threadId; pos < HISTO_HEIGHT * HISTO_WIDTH; pos += numThreadsBlock){
        int res = atomicAdd(result_array + pos, blockSharedHist[pos]);
        // if(globalTid == 1){
        //     printf("added to global: %d\n", res);
        // }
        atomicMin(result_array + pos, 255); // clamp to 255
    }
    // if(globalTid == 1){

    //     printf("print results array input: \n");
    //     for(int i = 0; i < HISTO_HEIGHT * HISTO_WIDTH; i++){
    //         printf("%d ", result_array[i]);
    //     }
    //     printf("\n\n");

    // }
}

void opt_2dhisto(uint32_t *result_array, uint32_t *input_array, size_t inputWidth)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
    // kernel invocation
    cudaMemset(result_array, 0, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint32_t));
	dim3 dimBlock(16, 16); // 256
	dim3 dimGrid((inputWidth + 16 - 1) / 16, (INPUT_HEIGHT + 16 - 1) / 16); // floor substitute ??
	HistogramKernel<<<dimGrid, dimBlock>>>(result_array, input_array, inputWidth);

}

/* Include below the implementation of any other functions you need */

void *AllocateDeviceMem(const void *src, size_t numBytes){
    uint32_t *destPtr = NULL;
    cudaMalloc((void**)&destPtr, numBytes);
    cudaMemcpy(destPtr, src, numBytes, cudaMemcpyHostToDevice);
    return destPtr;
}
void SynchronizeDeviceHost(){
    cudaDeviceSynchronize();
}
void CopyToHost(void *src, void *dest, size_t numBytes){
    cudaMemcpy(dest, src, numBytes, cudaMemcpyDeviceToHost);
}
void DeallocateMemory(void *src){
    cudaFree(src);
}

