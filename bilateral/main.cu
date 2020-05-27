#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <device_functions.h> 
#include "device_launch_parameters.h"
#include <cstdlib>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "bmp/EasyBMP.h"
#include <stdio.h>
#include <time.h>

using namespace std;
#define BLOCK_SIZE 16

float cpuGaussian[64];

__constant__ float cGaussian[64];

// объявляем ссылку на текстуру для двумерной текстуры float
texture<float, cudaTextureType2D, cudaReadModeElementType> textur;

__global__ void GPU_Bilateral(float * input, float *output, int width, int height,int radius, double euclidean_delta)
{
	int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

	double t = 0;
	double sumFactor = 0;
	unsigned char center = tex2D(textur, x, y);

	if ((x < width) && (y < height)) {

		for (int i = -radius; i <= radius; i++) {
			for (int j = -radius; j <= radius; j++) {

				unsigned char curPix = tex2D(textur, x + j, y + i);

				double factor = (cGaussian[i + radius] * cGaussian[j + radius]) * __expf(-(powf(center - curPix, 2)) / (2 * powf(euclidean_delta, 2)));
				t += factor * curPix;
				sumFactor += factor;
			}
		}
		output[y*width + x] = t / sumFactor;
	}
	
}


void writeImage(char *filePath, float *grayscale, unsigned int rows, unsigned int cols) {
	BMP Output;
	Output.SetSize(cols, rows);
	// записали картинку 
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			RGBApixel pixel;
			pixel.Red = grayscale[i * cols + j];
			pixel.Green = grayscale[i * cols + j];
			pixel.Blue = grayscale[i * cols + j];
			pixel.Alpha = 0;
			Output.SetPixel(j, i, pixel);
		}
	}
	Output.WriteToFile(filePath);
}

float *readImageGPU(char *filePathInput, unsigned int *rows, unsigned int *cols) {
	BMP Image;
	Image.ReadFromFile(filePathInput);
	*rows = Image.TellHeight();
	*cols = Image.TellWidth();
	float *imageAsArray = (float *)calloc(*rows * *cols, sizeof(float));
	// Преобразуем картику в черно-белую
	for (int i = 0; i < Image.TellWidth(); i++) {
		for (int j = 0; j < Image.TellHeight(); j++) {
			double Temp = 0.30*(Image(i, j)->Red) + 0.59*(Image(i, j)->Green) + 0.11*(Image(i, j)->Blue);
			Image(i, j)->Red = (unsigned char)Temp;
			Image(i, j)->Green = (unsigned char)Temp;
			Image(i, j)->Blue = (unsigned char)Temp;
			imageAsArray[j * *cols + i] = Temp;
		}
	}
	return imageAsArray;
}


void CPU_Bilateral(float *input, float *output, float euclidean_delta, int width, int height, int radius) {

	for (int i = 0; i < 2 * radius + 1; i++) {
		int x = i - radius;
		cpuGaussian[i] = exp(-(x * x) / (2 * euclidean_delta * euclidean_delta));
	}

	float domainDist, colorDist, factor;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float t = 0.0f;
			float sum = 0.0f;

			for (int i = -radius; i <= radius; i++) {
				int neighborY = y + i;
				if (neighborY < 0) {
					neighborY = 0;
				}
				else if (neighborY >= height) {
					neighborY = height - 1;
				}
				for (int j = -radius; j <= radius; j++) {
					domainDist = cpuGaussian[radius + i] * cpuGaussian[radius + j];

					int neighborX = x + j;

					if (neighborX < 0) {
						neighborX = 0;
					}
					else if (neighborX >= width) {
						neighborX = width - 1;
					}
					colorDist = exp(-(pow((input[neighborY * width + neighborX] - input[y * width + x]), 2)) / (2 * pow(euclidean_delta, 2))); 
					factor = domainDist * colorDist;
					sum += factor;
					t += factor * input[neighborY * width + neighborX];
				}
			}
			output[y * width + x] = t / sum;
		}
	}
}

int main() {
	unsigned int rows = 0;
	unsigned int cols = 0;
	int filter_radius = 1;
	float euclidean_delta = 10000;
	// считали картинку 
	float * imageAsArrayCPU = readImageGPU("woman.bmp", &rows, &cols);
	float *outImageCPU = (float *)calloc(rows*cols, sizeof(float));

	clock_t start_s = clock();
	CPU_Bilateral(imageAsArrayCPU, outImageCPU, euclidean_delta, rows, cols, filter_radius);
	clock_t stop_s = clock();
	cout << "Time on CPU: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << " milliseconds" << endl;
	writeImage("womanCPU.bmp", outImageCPU, rows, cols);

	float * imageAsArray = readImageGPU("woman.bmp", &rows, &cols);
	float *outImage = (float *)calloc(rows*cols, sizeof(float));

	cudaEvent_t start;
	cudaEventCreate(&start);

	cudaEvent_t stop;
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//Создали дескриптор канала с форматом Float
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,cudaChannelFormatKindFloat);
	cudaArray *cuArray;
	cudaMallocArray(&cuArray, &channelDesc, cols, rows);

	// Скопировали массив imageAsArray в cuArray
	cudaMemcpyToArray(cuArray, 0, 0, imageAsArray, rows * cols * sizeof(float),cudaMemcpyHostToDevice);


	float fGaussian[64];
	for (int i = 0; i < 2 * filter_radius + 1; i++)
	{
		float x = i - filter_radius;
		fGaussian[i] = expf(-(x*x) / (2 * euclidean_delta*euclidean_delta));
	}
	cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2 * filter_radius + 1));

	// Установили параметры текстуры
	textur.addressMode[0] = cudaAddressModeClamp;
	textur.addressMode[1] = cudaAddressModeClamp;
	textur.filterMode = cudaFilterModePoint;

	// Привязали массив к текстуре
	cudaBindTextureToArray(textur, cuArray, channelDesc);

	float *dev_grayscale, *dev_output, *output;
	output = (float *)calloc(rows * cols, sizeof(float));
	cudaMalloc(&dev_output, rows * cols * sizeof(float));
	cudaMalloc(&dev_grayscale, rows * cols * sizeof(float));

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x,(rows + dimBlock.y - 1) / dimBlock.y);

	GPU_Bilateral << <dimGrid, dimBlock >> >(dev_grayscale, dev_output, cols, rows, filter_radius, euclidean_delta);
	cudaMemcpy(output, dev_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);
	float msec = msecTotal;

	cout << "Time on GPU: " << msec << " milliseconds" << endl;
	writeImage("womanGPU.bmp", output, rows, cols);

	//Чистим ресурсы на GPU
	cudaFreeArray(cuArray);
	cudaFree(dev_output);

	system("pause");
	return 0;
}