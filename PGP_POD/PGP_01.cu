#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

__global__ void Kernel(double* a, double* b, int sz)
{
	int idx = (blockDim.x)*(blockIdx.x) + threadIdx.x;
	int offset = blockDim.x*gridDim.x;
	for (int i = idx; i < sz; i += offset) {
		a[i] = fmax(a[i],b[i]);

	}
}


int main() {
	int i, sz;
	scanf("%d", &sz);
	double* a = (double*)malloc(sizeof(double)*sz);
	double* b = (double*)malloc(sizeof(double)*sz);
	double* dev_a;
	double* dev_b;
	CSC(cudaMalloc(&dev_a, sizeof(double)*sz));
	CSC(cudaMalloc(&dev_b, sizeof(double)*sz));
	for (i = 0; i < sz; ++i) {
		scanf("%lf", &a[i]);
	}
	for (i = 0; i < sz; ++i) {
		scanf("%lf", &b[i]);
	}
	CSC(cudaMemcpy(dev_a, a, sizeof(double)*sz, cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(dev_b, b, sizeof(double)*sz, cudaMemcpyHostToDevice));
	
	Kernel <<<256, 256 >>> (dev_a, dev_b, sz);

	CSC(cudaMemcpy(a, dev_a, sizeof(double)*sz, cudaMemcpyDeviceToHost));
	for (i = 0; i < sz; ++i) {
		printf("%.10e\n", a[i]);
	}
	free(a);
	free(b);
	cudaFree(dev_a);
	cudaFree(dev_b);
	return 0;
}
