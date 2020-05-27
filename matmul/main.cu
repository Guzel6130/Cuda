
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>



//�������� ������ �� GPU
__global__ void Multiply_Matrix_GPU(float* A, float* B, float* C , int BLOCK_SIZE , int N) {
	// ������ �����
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// ������ ����
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float total = 0.0;
	int ia = N * BLOCK_SIZE * by + N * ty;
	int ib = BLOCK_SIZE * bx + tx;


	for (int k = 0; k < N; k++) {
		total += A[ia + k] * B[ib + k * N];
	}
	int ic = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;

	//�������������� �������
	C[ic + N * ty + tx] = total;
}



//��������� ������ �� CPU

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

	//�������� ������ ��� ������ ������ �� CPU
	float *A = (float*) malloc(N * N *sizeof(float));
	float *B = (float*) malloc(N * N* sizeof(float));
	float *C_GPU = (float*) malloc(N * N *sizeof(float));
	float *C_CPU = (float*) malloc(N * N*  sizeof(float));
	
	// ��������� ������� 
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++) {
			//�� -20 �� 20 ��������� �������
			A[i + j * N] = (int)rand() % 41 - 20;
			B[i + j * N] = (int)rand() % 41 - 20;
		}


	//������������ ������� ����
	dim3 dimGrid(N / BLOCK_SIZE, N / BLOCK_SIZE);//������ �����
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);//������ �����

	
	//event'� ��� ������ ������� ������ GPU
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	
	//��������  ������ ��� ������ ������ �� GPU
	float* adev, *bdev, *cdev;
	cudaMalloc((void**)&adev, N * N * sizeof(float *));
	cudaMalloc((void**)&bdev, N * N * sizeof(float *));
	cudaMalloc((void**)&cdev, N * N * sizeof(float *));

	//�������� �������� ������� � CPU �� GPU
	cudaMemcpy(adev, A, N * N * sizeof(float *), cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, B, N * N * sizeof(float *), cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);   					
	//��������� ������ �� GPU
	Multiply_Matrix_GPU << < dimGrid, dimBlock >> > (adev, bdev , cdev,BLOCK_SIZE,N);
	cudaEventRecord(stop, 0);    

	//��������������� � �������� ��������� ��������
	cudaEventSynchronize(stop);   

	//������������ ����� ������ GPU
	float timeGPU = 0;
	cudaEventElapsedTime(&timeGPU, start, stop);    
	std::cout << "GPU time: " << timeGPU << std::endl;

	//�������� ��������� � GPU �� CPU
	cudaMemcpy(C_GPU, cdev, N * N * sizeof(float *), cudaMemcpyDeviceToHost);

	//������ ������ �� GPU 
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(adev);
	cudaFree(bdev);
	cudaFree(cdev);

	double start_time = clock();

	//��������� ������ �� GPU

	Multiply_Matrix_CPU(A, B, C_CPU,N);

	double end_time = clock();

	std::cout << "CPU time  " << ((end_time - start_time)) *1000 / CLOCKS_PER_SEC << std::endl;
	
	//������ ������ �� CPU
	delete A;
	delete B;
	delete C_GPU;
	delete C_CPU;
	system("pause");
	return 0;
}