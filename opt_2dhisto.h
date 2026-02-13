#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t *result_array, uint32_t *input_array, size_t inputWidth);

/* Include below the function headers of any other functions that you implement */
void *AllocateDeviceMem(const void *src, size_t numBytes);
void SynchronizeDeviceHost();
void CopyToHost(void *src, void *dest, size_t numBytes);
void DeallocateMemory(void *src);
#endif
