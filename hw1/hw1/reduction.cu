#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <algorithm>
#include <ctime>
#include <cstdlib>

__global__ void min_kernel(float* inData, float* outData, int size)
{
	extern __shared__ float data[];

	int tid = threadIdx.x; //id thread
	int i = blockIdx.x * blockDim.x + threadIdx.x; // номер текущей нити
	if (i < size)
	{
		data[tid] = inData[i]; //load data from global memory to shared
		__syncthreads();

		for (int s = 1; s < blockDim.x; s *= 2)
		{
			if (tid % (2 * s) == 0 && i + s < size)
			{
				if (data[tid + s] < data[tid])
				{
					data[tid] = data[tid + s];
				}
			}
			__syncthreads();
		}

		//write result of block reduction to global memory
		if (tid == 0)
		{
			outData[blockIdx.x] = data[0];
		}
	}
}

const int ARRAY_SIZE = 10000;
const int SIZE = ARRAY_SIZE * sizeof(float);

const int BLOCK_SIZE = 512;
int GRID_SIZE = (ARRAY_SIZE % BLOCK_SIZE == 0) ? ARRAY_SIZE / BLOCK_SIZE : (ARRAY_SIZE / BLOCK_SIZE + 1);

float* host_ARRAY;
float* device_ARRAY;
float* host_result;
float* device_result;
cudaError_t err;

void initialize_array();
void malloc_cpu_memory();
void malloc_gpu_memory();
void copy_from_cpu_to_gpu();
void copy_from_gpu_to_cpu();
float min_cpu();
void free_all_memory();

int main()
{
	printf("Reduction task\n");

	malloc_cpu_memory();
	initialize_array();

	malloc_gpu_memory();
	copy_from_cpu_to_gpu();

	//измерение времени работы ядра
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float cuda_time;
	cudaEventRecord(start, 0);

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(GRID_SIZE, 1, 1);

	printf("CUDA kernel launch with %d blocks of %d threads\n", GRID_SIZE, BLOCK_SIZE);
	min_kernel << <dimGrid, dimBlock, BLOCK_SIZE * sizeof(float) >> >(device_ARRAY, device_result, ARRAY_SIZE);
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cuda_time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	copy_from_gpu_to_cpu();

	float gpu_result = INT_MAX;
	for (int i = 0; i < GRID_SIZE; i++)
	{
		gpu_result = std::min(gpu_result, host_result[i]);
	}

	int start_time = clock();
	float cpu_result = min_cpu();
	int end_time = clock();
	float cpu_time = (end_time - start_time) / 1000.0;

	printf("CUDA result is %f and CPU result is %f \n", gpu_result, cpu_result);
	printf("CUDA's time is: %f\n", cuda_time);
	printf("CPU's time is: %f\n", cpu_time);

	free_all_memory();

	printf("Done\n");
	return 0;
}

void initialize_array()
{
	srand(time(NULL));
	for (int i = 0; i < ARRAY_SIZE; i++) // Initialize the host input vectors
	{
		host_ARRAY[i] = rand() / (float)RAND_MAX;
	}
}

void malloc_cpu_memory()
{
	host_ARRAY = (float *)malloc(SIZE);

	if (host_ARRAY == 0)
	{
		fprintf(stderr, "Failed to allocate host memory for ARRAY!\n");
		exit(EXIT_FAILURE);
	}

	host_result = (float *)malloc(GRID_SIZE * sizeof(float));

	if (host_result == 0)
	{
		fprintf(stderr, "Failed to allocate host memory for result!\n");
		exit(EXIT_FAILURE);
	}
}

void malloc_gpu_memory()
{
	err = cudaMalloc((void **)&device_ARRAY, SIZE);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device memory for ARRAY (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&device_result, GRID_SIZE * sizeof(float));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device memory for result(error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void copy_from_cpu_to_gpu()
{
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(device_ARRAY, host_ARRAY, SIZE, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy ARRAY from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void copy_from_gpu_to_cpu()
{
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(host_result, device_result, GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy result from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void free_all_memory()
{
	cudaFree(device_ARRAY);
	cudaFree(device_result);

	free(host_ARRAY);
	free(host_result);
}

float min_cpu()
{
	float local_min = INT_MAX;
	for (int i = 0; i < ARRAY_SIZE; i++)
	{
		local_min = std::min(host_ARRAY[i], local_min);
	}
	return local_min;
}
