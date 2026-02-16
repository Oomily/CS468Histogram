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
    __shared__ uint32_t blockSharedHist[HISTO_HEIGHT * HISTO_WIDTH * 8]; // 8 histograms, one per warp (8 warps)

    for(int pos = threadId; pos < HISTO_HEIGHT * HISTO_WIDTH * 8; pos += numThreadsBlock){
        blockSharedHist[pos] = 0;
    }

    __syncthreads();

    // read from input and update block histogram
    for(int pos = globalTid; pos < INPUT_HEIGHT * inputWidth; pos += numThreadsTotal){
        int col = pos % inputWidth;
        if(col < INPUT_WIDTH) {// overflow threads on edges
            uint32_t data = input_array[pos];
            int warpNum = threadId / 1024;
            if(blockSharedHist[data + warpNum * 1024] < 255){ // try to limit unecessary atomicAdds
                uint32_t test = atomicAdd(blockSharedHist + data + warpNum * 1024, 1);
                if(test >= 255){ // must test to not over add
                    blockSharedHist[data + warpNum * 1024] = 255;
                }
            }
        }
    }
    __syncthreads();

    for(int pos = threadId; pos < HISTO_HEIGHT * HISTO_WIDTH * 8; pos += numThreadsBlock){
        int warpNum = pos / 1024;
        int res = atomicAdd(result_array + pos - warpNum * 1024, blockSharedHist[pos]);
        atomicMin(result_array + pos - warpNum * 1024, 255); // clamp to 255
    }
}

void opt_2dhisto(uint32_t *result_array, uint32_t *input_array, size_t inputWidth)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
    // kernel invocation
    cudaMemset(result_array, 0, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint32_t));
	dim3 dimBlock(32, 8);
	// dim3 dimGrid((inputWidth + 32 - 1) / 32, (INPUT_HEIGHT + 8 - 1) / 8); // floor substitute ??
	dim3 dimGrid(8, 8);

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
