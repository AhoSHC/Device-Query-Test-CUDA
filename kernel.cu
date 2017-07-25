#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define DATA_SIZE 1048576
int data[DATA_SIZE];

void GenerateNumbers(int *number, int size)
{
	for (int i = 0; i < size; i++)
	{
		number[i] = rand() % 10;
	}
}

__global__ static void sumOfSquares(int *num, int* result)
{
	int sum = 0;
	int i;

	for (i = 0; i < DATA_SIZE; i++)
	{
		sum += num[i] * num[i];
	}
	*result = sum;

}

// output given cudaDeviceProp
void OutputSpec(const cudaDeviceProp sDevProp)
{
	printf("1. 在initCUDA中讀出\n");
	printf("1). Device name: %s\n", sDevProp.name);
	printf("2). Total amount of global memory: %.0f MBytes (%llu bytes)\n",
		(float)sDevProp.totalGlobalMem / DATA_SIZE, (unsigned long long) sDevProp.totalGlobalMem);
	printf("3). Maximum number of threads per multiprocessor:  %d\n", sDevProp.maxThreadsPerMultiProcessor);
	printf("3). Maximum number of threads per block:           %d\n", sDevProp.maxThreadsPerBlock);
	printf("3). Max dimension size of a thread block (x,y,z): ( %d, %d, %d )\n", sDevProp.maxThreadsDim[0], sDevProp.maxThreadsDim[1], sDevProp.maxThreadsDim[2]);
	printf("4). GPU Max Clock rate: %.0f MHz (%0.2f GHz)\n\n", sDevProp.clockRate * 1e-3f, sDevProp.clockRate * 1e-6f);
}


void main()
{
	// part1, check the number of device
	int  iDeviceCount = 0;
	cudaGetDeviceCount(&iDeviceCount);
	//printf("Number of GPU: %d\n", iDeviceCount);

	if (iDeviceCount == 0)
	{
		printf("No supported GPU\n");
		return;
	}

	// part2, output information of each device
	for (int i = 0; i < iDeviceCount; ++i)
	{
		//printf("\n=== Device %i ===\n", i);
		cudaDeviceProp  sDeviceProp;
		cudaGetDeviceProperties(&sDeviceProp, i);
		OutputSpec(sDeviceProp);
	}


	//run of GPU
	GenerateNumbers(data, DATA_SIZE);
	int* gpudata, *result, sum;

	cudaEvent_t go, stop;
	float time_gpu;
	cudaEventCreate(&go);
	cudaEventCreate(&stop);

	cudaMalloc((void**)&gpudata, sizeof(int) * DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int));

	cudaMemcpy(gpudata, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

	cudaEventRecord(go, 0);
	sumOfSquares << <1, 1, 0 >> >(gpudata, result);
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);

	cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(data, gpudata, sizeof(float)*DATA_SIZE, cudaMemcpyDeviceToHost);

	cudaFree(gpudata);
	cudaFree(result);
	cudaEventElapsedTime(&time_gpu, go, stop);
	time_gpu = time_gpu / 1000.0;

	printf("2. 在GPU和CPU上分別計算結果\n");
	printf("sum (GPU): %d\n", sum);
	//run of CPU
	int cpu_time = 0;
	sum = 0;
	unsigned long start = clock();
	for (int i = 0; i < DATA_SIZE; i++)
	{
		sum += data[i] * data[i];
	}
	unsigned long end = clock();
	
	printf("sum (CPU): %d\n\n", sum);
	printf("3. GPU和CPU計算執行時間\n");
	printf("time (GPU) %1.3fs\n",time_gpu);
	printf("time (CPU): %1.3fs\n\n", (end - start) / 1000.0);

}