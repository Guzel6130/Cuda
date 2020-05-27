
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>



//Уножение матриц на GPU
__global__ void Multiply_Matrix_GPU(float* A, float* B, float* C , int BLOCK_SIZE , int N) {
	// Индекс блока
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Индекс нити
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float total = 0.0;
	int ia = N * BLOCK_SIZE * by + N * ty;
	int ib = BLOCK_SIZE * bx + tx;


	for (int k = 0; k < N; k++) {
		total += A[ia + k] * B[ib + k * N];
	}
	int ic = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;

	//Результирующая матрица
	C[ic + N * ty + tx] = total;
}



//Умножение матриц на CPU

void Multiply_Matrix_CPU(float* A, float* B, float* C, int N) {
	for (int i = 0; i < N; i++) {
		for (int k = 0; k < N; k++) {
			for (int j = 0; j < N; j++) {
				C[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}

int main() {
	const int  BLOCK_SIZE = 16;
	const int N = 2048;

	//Выделяем память для храния данных на CPU
	float *A = (float*) malloc(N * N *sizeof(float));
	float *B = (float*) malloc(N * N* sizeof(float));
	float *C_GPU = (float*) malloc(N * N *sizeof(float));
	float *C_CPU = (float*) malloc(N * N*  sizeof(float));
	
	// Заполняем матрицы 
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
			//от -20 до 20 заполнили матрицы
			A[i + j * N] = (int)rand() % 41 - 20;
			B[i + j * N] = (int)rand() % 41 - 20;
		}


	//Конфигурация запуска ядра
	dim3 dimGrid(N / BLOCK_SIZE, N / BLOCK_SIZE);//Размер сетки
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);//Размер блока

	
	//event'ы для замера времени работы GPU
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	
	//Выделяем  память для храния данных на GPU
	float* adev, *bdev, *cdev;
	cudaMalloc((void**)&adev, N * N * sizeof(float *));
	cudaMalloc((void**)&bdev, N * N * sizeof(float *));
	cudaMalloc((void**)&cdev, N * N * sizeof(float *));

	//Копируем исходные матрицы с CPU на GPU
	cudaMemcpy(adev, A, N * N * sizeof(float *), cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, B, N * N * sizeof(float *), cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);   					
	//Умножение матриц на GPU
	Multiply_Matrix_GPU << < dimGrid, dimBlock >> > (adev, bdev , cdev,BLOCK_SIZE,N);
	cudaEventRecord(stop, 0);    

	//Синхронизируемя с моментом окончания расчетов
	cudaEventSynchronize(stop);   

	//Рассчитываем время работы GPU
	float timeGPU = 0;
	cudaEventElapsedTime(&timeGPU, start, stop);    
	std::cout << "GPU time: " << timeGPU << std::endl;

	//Копируем результат с GPU на CPU
	cudaMemcpy(C_GPU, cdev, N * N * sizeof(float *), cudaMemcpyDeviceToHost);

	//Чистим память на GPU 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(adev);
	cudaFree(bdev);
	cudaFree(cdev);

	double start_time = clock();

	//Умножение матриц на GPU

	Multiply_Matrix_CPU(A, B, C_CPU,N);

	double end_time = clock();

	std::cout << "CPU time  " << ((end_time - start_time)) *1000 / CLOCKS_PER_SEC << std::endl;
	
	//Чистим память на CPU
	delete A;
	delete B;
	delete C_GPU;
	delete C_CPU;
	system("pause");
	return 0;
}