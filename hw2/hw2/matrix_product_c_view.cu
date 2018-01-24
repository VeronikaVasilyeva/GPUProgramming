#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <thrust/device_vector.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <time.h>

// C-style indexing
int ci(int row, int column, int nColumns)
{
	return row * nColumns + column;
}

float* h_A;
float* h_B;
float* h_C;

int rowA = 10;
int colA = 8;

int rowB = colA;
int colB = 2;

int rowC = rowA;
int colC = colB;

cublasHandle_t handle;
cublasStatus_t status;

void device_info();
void gpu_random_init(float* A, int rows, int cols);

//void copy_from_gpu_to_cpu(float* A, float* B);

void free_all_memory();

void gpu_matrix_product(const float* A, const float* B, float* C, const int m, const int k, const int n);
void cpu_matrix_product();

void matrix_print(float* A, int rows, int cols);

int main()
{
	device_info();

	// allocate three device_vectors with row*col elements
	thrust::device_vector<float> d_A(rowA * colA);
	thrust::device_vector<float> d_B(rowB * colB);
	thrust::device_vector<float> d_C(rowC * colC);

//	gpu_random_init(raw_pointer_cast(&d_A[0]), rowA, colA);
	//gpu_random_init(raw_pointer_cast(&d_B[0]), rowB, colB);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	//gpu_matrix_product(raw_pointer_cast(&d_A[0]), raw_pointer_cast(&d_B[0]), raw_pointer_cast(&d_C[0]), rowA, colA, colB);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("CUBLAS time : %f ms\n", milliseconds);
	
	//matrix_print(raw_pointer_cast(&d_A[0]), rowA, colA);
	//matrix_print(raw_pointer_cast(&d_B[0]), rowB, colB);
	//matrix_print(raw_pointer_cast(&d_C[0]), rowC, colC);

//	copy_from_gpu_to_cpu(raw_pointer_cast(&d_A[0]), raw_pointer_cast(&d_B[0]));
	
	status = cublasDestroy(handle);
	if (status != CUBLAS_STATUS_SUCCESS)
		printf("shutdown error %s \n", status);

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

//void copy_from_gpu_to_cpu(float* A, float* B)
//{
//	h_A = (float *)malloc(rowA * colA * sizeof(float)); 
//	h_B = (float *)malloc(rowB * colB * sizeof(float)); 
//	h_C = (float *)malloc(rowC * colC * sizeof(float));
//
//	cudaMemcpy(h_A, A, rowA * colA * sizeof(float), cudaMemcpyDeviceToHost);
//	cudaMemcpy(h_B, B, rowB * colB * sizeof(float), cudaMemcpyDeviceToHost);
//}

// C(m,n) = A(m,k) * B(k,n)
void gpu_matrix_product(const float* A, const float* B, float* C, const int m, const int k, const int n)
{
	// Initialize CUBLAS 
	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
		printf("CUBLAS initialization error with message %s\n", status);

	float alpha = 1.0f;
	float beta = 0.0f;

	//C = alpha*A*B + beta * C
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, colB, rowA, colA, &alpha, B, colB, A, colA, &beta, C, colC);

	if (status != CUBLAS_STATUS_SUCCESS)
		printf("Kernel execution error with message %s\n", status);
}

void cpu_matrix_product()
{

}

void gpu_random_init(float* A, int rows, int cols)
{
	// Create a pseudo-random number generator
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock());

	// Fill the array with random numbers on the device
	size_t n = rows * cols;
	curandGenerateUniform(gen, A, n);
	curandDestroyGenerator(gen); /* Cleanup */
}

void matrix_print(float* A, int rows, int cols)
{
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			std::cout << A[ci(i, j, cols)] << " ";
		}
		printf("\n");
	}
}
