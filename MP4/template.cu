#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
const int MASK_RADIUS = 1;
const int MASK_SIZE = 3;
const int TILED_SIZE = 3;
const int CACHE_SIZE = TILED_SIZE + MASK_SIZE - 1;
//@@ Define constant memory for device kernel here
__constant__ float MASK[27];
__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int bz = blockIdx.z;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tz = threadIdx.z;
  __shared__ float matrix[CACHE_SIZE][CACHE_SIZE][CACHE_SIZE];

  int output_x = bx * TILED_SIZE + tx;
  int output_y = by * TILED_SIZE + ty;
  int output_z = bz * TILED_SIZE + tz;

  int input_x = output_x - MASK_RADIUS;
  int input_y = output_y - MASK_RADIUS;
  int input_z = output_z - MASK_RADIUS;

  if (input_x >= 0 && input_x < x_size && input_y >= 0 && input_y < y_size &&
      input_z >= 0 && input_z < z_size) {
    matrix[tz][ty][tx] =
        input[input_z * y_size * x_size + input_y * x_size + input_x];
  } else {
    matrix[tz][ty][tx] = 0.0;
  }

  __syncthreads();

  float value = 0.0;
  if (tx < TILED_SIZE && ty < TILED_SIZE && tz < TILED_SIZE) {
    for (int i = 0; i < MASK_SIZE; i++) {
      for (int j = 0; j < MASK_SIZE; j++) {
        for (int k = 0; k < MASK_SIZE; k++) {
          value +=
              MASK[i * TILED_SIZE * TILED_SIZE + j * TILED_SIZE + k] *
              matrix[tz + i][ty + j][tx + k];
        }
      }
    }
    if (output_x < x_size && output_y < y_size && output_z < z_size)
      output[output_z * x_size * y_size + output_y * x_size + output_x] = value;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
 
  
  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**) &deviceInput, sizeof(float) * (inputLength-3));
  cudaMalloc((void**) &deviceOutput, sizeof(float) * (inputLength-3));
  wbTime_stop(GPU, "Doing GPU memory allocation");
  
  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput+3, sizeof(float) * z_size * y_size * x_size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(MASK, hostKernel, sizeof(float) * kernelLength, 0, cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil((1.0 * x_size)/TILED_SIZE), ceil((1.0 * y_size)/TILED_SIZE), ceil((1.0 * z_size)/TILED_SIZE));
  dim3 DimBlock(CACHE_SIZE, CACHE_SIZE, CACHE_SIZE);
  conv3d <<< DimGrid, DimBlock >>> (deviceInput, deviceOutput, z_size, y_size, x_size);
  //@@ Launch the GPU kernel here
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy((hostOutput+3), deviceOutput, sizeof(float) * z_size * y_size * x_size, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}