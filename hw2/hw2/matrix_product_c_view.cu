#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <thrust/device_vector.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <iostream>
#include <cuda.h>
#include <stdio.h>

// C-style indexing
int ci(int row, int column, int nColumns)
{
	return row * nColumns + column;
}

const int MATRIX_SIZE = 10;
//const int SIZE = ARRAY_SIZE * sizeof(float);

cudaError_t err;
const int BLOCK_SIZE = 512;
//int GRID_SIZE = (ARRAY_SIZE % BLOCK_SIZE == 0) ? ARRAY_SIZE / BLOCK_SIZE : (ARRAY_SIZE / BLOCK_SIZE + 1);

float* h_A;
float* h_B;
float* d_A;
float* d_B;
float* h_C;
float* d_C;

void device_info();
void initialize_matrix();
void malloc_cpu_memory();
void malloc_gpu_memory();
void copy_from_cpu_to_gpu();
void copy_from_gpu_to_cpu();
void free_all_memory();
float matrix_product_cpu();

int main()
{
	device_info();

	// initialize data
	thrust::device_vector<float> D(MATRIX_SIZE * MATRIX_SIZE);
	thrust::device_vector<float> E(MATRIX_SIZE * MATRIX_SIZE);
	thrust::device_vector<float> F(MATRIX_SIZE * MATRIX_SIZE);

	for (size_t i = 0; i < MATRIX_SIZE; i++)
	{
		for (size_t j = 0; j < MATRIX_SIZE; j++)
		{
			D[ci(i, j, MATRIX_SIZE)] = i + j;
			std::cout << D[ci(i, j, MATRIX_SIZE)] << " ";
		}
		printf("\n");
	}

	for (size_t i = 0; i < MATRIX_SIZE; i++)
	{
		for (size_t j = 0; j < MATRIX_SIZE; j++)
		{
			E[ci(i, j, MATRIX_SIZE)] = i + j;
			std::cout << E[ci(i, j, MATRIX_SIZE)] << " ";
		}
		printf("\n");
	}

	for (size_t i = 0; i < MATRIX_SIZE; i++)
	{
		for (size_t j = 0; j < MATRIX_SIZE; j++)
		{
			F[ci(i, j, MATRIX_SIZE)] = 0;
		}
	}
	cublasHandle_t handle;

	/* Initialize CUBLAS */
	cublasStatus_t status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printf("CUBLAS initialization error with message %s\n", status);
	}

	float alpha = 1.0f;
	float beta = 0.0f;
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
	                     MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE,
	                     &alpha, raw_pointer_cast(&E[0]), MATRIX_SIZE,
	                     raw_pointer_cast(&D[0]), MATRIX_SIZE,
	                     &beta, raw_pointer_cast(&F[0]), MATRIX_SIZE); // colE  x rowD
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printf("Kernel execution error with message %s\n", status);
	}

	for (size_t i = 0; i < MATRIX_SIZE; i++)
	{
		for (size_t j = 0; j < MATRIX_SIZE; j++)
		{
			std::cout << F[ci(i, j, MATRIX_SIZE)] << " ";
		}
		printf("\n");
	}

	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printf("shutdown error %s \n", status);
	}

	return 0;
}


void device_info()
{
	int kb = 1024;
	int mb = kb * kb;

	int GPU_N;
	cudaGetDeviceCount(&GPU_N);
	printf("Device count: %d\n", GPU_N);

	for (int i = 0; i < GPU_N; i++)
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		printf("PCI Bus id: %d\n", props.pciBusID);

		cudaGetDeviceProperties(&props, i);
		printf("Device %i: %s: %i.%i\n", i, props.name, props.major, props.minor);
		printf("Global memory: %i mb\n", props.totalGlobalMem / mb);
		printf("Shared memory: %i kb\n", props.sharedMemPerBlock / kb);
		printf("Constant memory:  %i kb\n", props.totalConstMem / kb);
		printf("Block registers: %i\n", props.regsPerBlock);
		printf("Warp size: %i\n", props.warpSize);
		printf("Threads per block: %i\n", props.maxThreadsPerBlock);
		printf("Max block dimensions: [ %i, %i, %i]\n", props.maxThreadsDim[0], props.maxThreadsDim[1],
		       props.maxThreadsDim[2]);
		printf("Max grid dimensions:  [ %i, %i, %i]\n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
	}
}
