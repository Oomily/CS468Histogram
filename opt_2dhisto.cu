#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

__global__ void HistogramKernel(uint32_t *result_array, uint32_t *input_array, size_t inputWidth)
{
    const int numThreadsBlock = blockDim.x * blockDim.y;
    const int threadId = threadIdx.y * blockDim.x + threadIdx.x; // flattened within a block
    __shared__ uint32_t blockSharedHist[HISTO_HEIGHT * HISTO_WIDTH];
    __shared__ uint32_t blockInputArray[32 * 32];

    for(int pos = threadId; pos < HISTO_HEIGHT * HISTO_WIDTH; pos += numThreadsBlock){
        blockSharedHist[pos] = 0;
    }

    // pull a tile of input array into shared memory
    blockInputArray[threadId] = input_array[(blockIdx.y * 32 + threadIdx.y) * inputWidth + blockIdx.x * 32 + threadIdx.x];
    __syncthreads();

    // read from input and update block histogram
    for(int pos = 0; pos < 32 * 32; pos++){
        int realcol = blockIdx.x * 32 + (pos % 32);
        if(realcol < INPUT_WIDTH) {// check for overflow threads on edges
            uint32_t data = blockInputArray[pos];
            if(data % (32*32) == threadId % (32*32)){
                if(blockSharedHist[data] < 255){ // try to limit unecessary atomicAdds
                    uint32_t test = atomicAdd(blockSharedHist + data, 1);
                    if(test >= 255){ // must test to not over add
                        blockSharedHist[data] = 255;
                    }
                }
            }
        }
    }
    __syncthreads();

    for(int pos = threadId; pos < HISTO_HEIGHT * HISTO_WIDTH; pos += numThreadsBlock){
        if(pos < HISTO_HEIGHT * HISTO_WIDTH){
            int res = atomicAdd(result_array + pos, blockSharedHist[pos]);
            atomicMin(result_array + pos, 255); // clamp to 255
        }
    }
}

void opt_2dhisto(uint32_t *result_array, uint32_t *input_array, size_t inputWidth)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
    // kernel invocation
    cudaMemset(result_array, 0, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint32_t));
	dim3 dimBlock(32, 32); // 1024 = 32 * 32, so 32 threads * 32 banks works out perfectly for columns. do how ever many rows for convenience
	dim3 dimGrid((inputWidth + 32 - 1) / 32, (INPUT_HEIGHT + 32 - 1) / 32); // floor substitute ??
	// dim3 dimGrid(1, 1);
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

