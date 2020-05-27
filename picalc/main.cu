#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <device_functions.h> 
#include "device_launch_parameters.h"
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand.h>
#include <math.h>

__global__ void Pi_GPU(float *x, float *y, int *totalCounts, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // номер элемента
	int threadCount = gridDim.x * blockDim.x; //cмещение

	int countPoints = 0;
	for (int i = idx; i < N; i += threadCount) {
		if (x[i] * x[i] + y[i] * y[i] < 1) {
			countPoints++;
		}
	}
	atomicAdd(totalCounts, countPoints); // каждый поток суммирует в переменную
}

float PI_CPU(float *x, float *y, int N) {
	int countPoints = 0; //Кол-во точек в круге
	for (int i = 0; i < N; i++) {
		if (x[i] * x[i] + y[i] * y[i] < 1) {
			countPoints++;
		}
	}
	return float(countPoints) * 4 / N;
}

int main(){ 
	// Количество точек 
	const long long N = 20000000;
	// Выделяем память для храния данных на CPU
	float *X, *Y, *devX, *devY;
	X = (float *)calloc(N, sizeof(float));
	Y = (float *)calloc(N, sizeof(float));

	//Выделяем память для храния данных на GPU
	cudaMalloc((void **)&devX, N * sizeof(float));
	cudaMalloc((void **)&devY, N * sizeof(float));

	//создаем новый генератор
	curandGenerator_t curandGenerator; 
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT); 
	curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL); 
	// генерируем числа
	curandGenerateUniform(curandGenerator, devX, N); 
	curandGenerateUniform(curandGenerator, devY, N);

	curandDestroyGenerator(curandGenerator); 

	//Копируем заполненные вектора с GPU на CPU
	cudaMemcpy(X, devX, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(Y, devY, N * sizeof(float), cudaMemcpyDeviceToHost);
	
	int blockDim = 512; 
	dim3 threads(blockDim, 1);
	dim3 grid(N / (128 * blockDim), 1);

	int *gpu_total_counts = 0;
	int*gpu_total_counts_host = (int *)calloc(1, sizeof(int));
	cudaMalloc((void **)&gpu_total_counts, 512 * sizeof(int));

	//Создаем event'ы для замера времени работы GPU
	float gpuTime = 0;

	cudaEvent_t start;
	cudaEvent_t stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//старт расчетов на GPU
	cudaEventRecord(start, 0);
	Pi_GPU << <grid, threads >> >(devX, devY, gpu_total_counts, N);
	//Копируем результат с GPU на CPU
	cudaMemcpy(gpu_total_counts_host, gpu_total_counts, sizeof(int), cudaMemcpyDeviceToHost);
	//число пи на GPU
	float gpu_result = (float) *gpu_total_counts_host * 4 / N;
	//Отмечаем окончание расчета
	cudaEventRecord(stop, 0);

	//Синхронизируемя с моментом окончания расчетов
	cudaEventSynchronize(stop);

	//Рассчитываем время работы GPU
	cudaEventElapsedTime(&gpuTime, start, stop);

	std::cout << "GPU time " << gpuTime << "  Result: " << gpu_result << std::endl;

	//Чистим ресурсы на GPU
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(devX);
	cudaFree(devY);
	cudaFree(gpu_total_counts);

	clock_t  start_time = clock();
	float cpu_result = PI_CPU(X, Y, N);
	clock_t  end_time = clock();
	std::cout << "CPU time " << (double)((end_time - start_time) * 1000 / CLOCKS_PER_SEC) << "  Result : " << cpu_result << std::endl;

	//Чистим память на CPU
	delete X;
	delete Y;
	return 0;
}
