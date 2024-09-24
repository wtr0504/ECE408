// MP 1
#include <wb.h>
__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  // int tid = blockDim.x * threadIdx.y + threadIdx.x;
  // int bid = gridDim.x * blockDim.y + blockDim.x;
  // int id = bid * (blockDim.x * blockDim.y) + tid;
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  for(int i = 0; id + i < len; i += gridDim.x * blockDim.x)
    out[id + i] = in1[id + i] + in2[id + i]; 
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceInput1,inputLength * sizeof(float));
  cudaMalloc((void**)&deviceInput2,inputLength * sizeof(float));
  cudaMalloc((void**)&deviceOutput,inputLength * sizeof(float));


  wbTime_stop(GPU, "Allocating GPU memory.");
  cudaMemcpy((void*)deviceInput1, (void*)hostInput1, inputLength*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy((void*)deviceInput2, (void*)hostInput2, inputLength*sizeof(float), cudaMemcpyHostToDevice);

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  dim3 blocks(2);
  dim3 threadsPerblock(32);
  vecAdd<<<blocks,threadsPerblock>>>(deviceInput1,deviceInput2,deviceOutput,inputLength);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy((void*)hostOutput, (void*)deviceOutput, inputLength*sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree( deviceOutput);
  cudaFree( deviceInput1);
  cudaFree( deviceInput2);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
