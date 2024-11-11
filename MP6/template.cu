// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

// #define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define BLOCK_SIZE 1024

#define BLOCK_NUM 1024
__device__ float blocks_max[BLOCK_NUM];
__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  const int idx = blockDim.x * blockIdx.x + threadIdx.x;
  const int tid = threadIdx.x;
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (idx < len && tid - stride >= 0) {
      output[idx] = input[idx] + input[idx - stride];
      __syncthreads();
      input[idx] = output[idx];
      __syncthreads();
    }
  }
  if (tid == blockDim.x - 1  && idx < len) {
    blocks_max[blockIdx.x] = output[idx];
    // printf("block idx : %d, block max %f\n",blockIdx.x,output[idx]);
  }
}

__global__ void add_blocks(float *in_out, int len) {
  for (int i = 0; i < blockIdx.x; i++) {
    if(blockDim.x * blockIdx.x + threadIdx.x < len) {
      in_out[blockIdx.x * blockDim.x + threadIdx.x] += blocks_max[i];
    }
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  // wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy((void*)deviceInput,(void*) hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy((void*)deviceOutput,(void*)hostInput,numElements * sizeof(float),cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  dim3 blocks((numElements + BLOCK_SIZE -1 / BLOCK_SIZE));
  dim3 threadsPerblock(BLOCK_SIZE);
  scan<<<blocks, threadsPerblock>>>(deviceInput, deviceOutput, numElements);
  wbCheck(cudaDeviceSynchronize());
  add_blocks<<<blocks, threadsPerblock>>>(deviceOutput, numElements);
  printf("---===-\n");
  wbCheck(cudaDeviceSynchronize());
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy((void*)hostOutput, (void*)deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
