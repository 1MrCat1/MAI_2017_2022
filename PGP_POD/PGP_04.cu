
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/extrema.h>
#include <thrust/device_vector.h> 
#include <stdio.h>
#include <stdlib.h>
#include <cmath>


#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)


__constant__ double ZERO = 1e-7;
//double ZERO = 10e-7;

struct nonZeroComparator
{
	__host__ __device__ bool operator()(double a) {
		return fabs(a) > ZERO;
	}
};

struct absComparator {
	__device__ bool operator()(double a, double b) {
		return fabs(a) < fabs(b);
	}
};

__host__ void Transpose(double* in, double* out,int w,int h) {
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			out[j*h+i] = in[i*w + j];
		}
	}
}

void printM(double* in, int w, int h) {
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			printf("%lf ", in[i*w + j]);
		}
		printf("\n");
	}
}

__global__ void swapColumns(double *data, int w,int h, int curCol, int newCol)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	for (int i = idx; i < h; i += offset) {
		double tmp = data[i * w + curCol];
		data[i * w + curCol] = data[i * w + newCol];
		data[i * w + newCol] = tmp;
	}
}

__global__ void findMultipliers(double* data, int w, int h, int row) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	for (int i = idx+row+1; i < w; i += offset) {
		data[row*w + i] /= data[row*w + row];
	}
}

__global__ void PM(double* data, int w, int h) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	for (int i = idx; i < w*h; i += offset) {
		printf("%lf ",data[i]);
	}
}

__global__ void substractColumns(double* data, int w, int h,int baseRow, int baseCol) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	for (int x = idx + baseCol + 1; x < w; x += offsetx) {
		for (int y = idy + baseRow + 1; y < h; y += offsety) {
			data[y*w + x] -= data[y*w + baseCol] * (data[baseRow*w + x]/data[baseRow*w+baseCol]);
		}
	}

}

int main(){
	int h, w; //n=h,m=w
	//FILE* fl = fopen("in.txt", "r");
	//fscanf(fl, "%d %d", &h,&w);
	scanf("%d %d", &h, &w);
	double* matrix = (double*)malloc(sizeof(double)*h*w);
	for (int i = 0; i < h; ++i) {
		for (int j = 0; j < w; ++j) {
			//fscanf(fl, "%lf ", &matrix[i]);
			scanf("%lf", &matrix[j*h+i]);
		}
	}

	int tmp = w;
	w = h;
	h = tmp;

	double* GPUdata;
	CSC(cudaMalloc(&GPUdata, sizeof(double)*h*w));
	CSC(cudaMemcpy(GPUdata, matrix, sizeof(double)*h*w, cudaMemcpyHostToDevice));
	nonZeroComparator nonZeroComp;
	absComparator absComp;
	thrust::device_ptr<double> curRow, nonZeroColumn;
	int rank = 0;
	//int curCol = 0;
	for (int i = 0; i < h; ++i) {
		curRow = thrust::device_pointer_cast(GPUdata + i * w);
		nonZeroColumn = thrust::max_element(curRow + rank, curRow + w, absComp);
		if (thrust::find_if(nonZeroColumn, nonZeroColumn+1, nonZeroComp)==nonZeroColumn+1) {
			continue;
		}
		if (curRow + rank != nonZeroColumn) {
			swapColumns << <256, 256 >> > (GPUdata, w, h, rank, nonZeroColumn - curRow);
		}
		//findMultipliers << <256, 256 >> > (GPUdata, w, h, i);
		substractColumns << <dim3(16, 16), dim3(16, 16) >> > (GPUdata, w, h, i, rank);
		++rank;
		if(rank == std::min(w,h)){
			break;
		}
	}
	cudaFree(GPUdata);
	fprintf(stderr, "INFO: H=%d, W=%d, rank=%d\n", h, w,rank);
	if(w*h<50){
		for(int i=0;i<h;++i){
			for(int j=0;j<w;++j){
			fprintf(stderr, "%.10e ",matrix[i*w+j]);
			}
			fprintf(stderr, "\n");
		}
	}
	free(matrix);
	printf("%d\n",rank);
	
}
//
	/*curRow = thrust::device_pointer_cast(GPUdata + total * w);
		++total;
		nonZeroColumn = thrust::max_element(curRow + i, curRow + w, absComp);
		if (thrust::find_if(nonZeroColumn, nonZeroColumn+1, nonZeroComp)==nonZeroColumn+1) {
			--i;
			if(total==h){
				break;
			}
			continue;
		}
		if (curRow + i != nonZeroColumn) {
			swapColumns << <256, 256 >> > (GPUdata, w, h, i, nonZeroColumn - curRow);
		}
		//findMultipliers << <256, 256 >> > (GPUdata, w, h, i);
		substractColumns << <dim3(16, 16), dim3(16, 16) >> > (GPUdata, w, h, i);
		++rank;
		if(total==h){
			break;
		}*/
