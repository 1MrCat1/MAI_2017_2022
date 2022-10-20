#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/extrema.h>
#include <thrust/device_vector.h> 
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <ctime>




#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

#define _sharedIdx(i) ((i)+(i)/31) 
#define BLOCK_SIZE 512
//#define THREADS_PER_BLOCK 64
#define SHARED_MEM_SIZE BLOCK_SIZE*2

#define DEBUG_PRINT false

bool CheckSorted(int* arr, int sz) {
	bool sorted = true;
	int counter = 0;
	for (int i = 0; i < sz-1; ++i) {
		if (arr[i] > arr[i + 1]) {
			if(sorted) fprintf(stderr,"SORTING ERROR.\n");
			fprintf(stderr, "%d %d:%d %d\n", i, i + 1, arr[i], arr[i + 1]);
			sorted = false;
			++counter;
		}
	}
	if (counter) fprintf(stderr, "Total errors: %d. Size: %d\n", counter, sz);
	return sorted;
}

__global__ void EvenOddPreSort(int* data, int sz) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int offset = blockDim.x*gridDim.x;
	int temp,id,i;
	//int oddoffset = idx / ((BLOCK_SIZE_G / 2) - 1);
	for (int k = 0; k < BLOCK_SIZE; ++k) {
		i = idx;
		id = 2 * i + k % 2;
		//id2 = 2 * i + 1 + k % 2;
		while ((id+1) < sz) {
			if ((k % 2 == 1) && ((id+1)%BLOCK_SIZE == 0)) {
				i += offset;
				id = 2 * i + k % 2;
				//id2 = 2 * i + 1 + k % 2;
				continue;
			}

			assert((id < sz) && (id+1 < sz));
			if (data[id] > data[id+1]) {
				temp = data[id];
				data[id] = data[id+1];
				data[id+1] = temp;
			}
			i += offset;
			id = 2 * i + k % 2;
			//id2 = 2 * i + 1 + k % 2;
		}
		//printf("Step:%d completed\n", k);
		__syncthreads();
	}
	//printf("%d:%d\n",idx, oddoffset);
}


__device__ void BitonicMerge(int* shMem) {
	int temp;
	//Первый проход инвертирован, т.к. у нас 2 изначально правильно отсортированных блока, а не битонич. послед.
	if (shMem[_sharedIdx(threadIdx.x)] > shMem[_sharedIdx(SHARED_MEM_SIZE - threadIdx.x - 1)]) {
		temp = shMem[_sharedIdx(threadIdx.x)];
		shMem[_sharedIdx(threadIdx.x)] = shMem[_sharedIdx(SHARED_MEM_SIZE - threadIdx.x - 1)];
		shMem[_sharedIdx(SHARED_MEM_SIZE - threadIdx.x - 1)] = temp;
	}

	__syncthreads();
	int ind;
	for (int binStep = (BLOCK_SIZE >> 1); binStep > 0; binStep >>= 1) {
		ind = threadIdx.x%binStep + 2*(binStep*(threadIdx.x / binStep));
		if (shMem[_sharedIdx(ind)] > shMem[_sharedIdx(ind + binStep)]) {
			temp = shMem[_sharedIdx(ind)];
			shMem[_sharedIdx(ind)] = shMem[_sharedIdx(ind + binStep)];
			shMem[_sharedIdx(ind + binStep)] = temp;
		}
		__syncthreads();
	}
}

__global__ void BitonicBlockMerge(int* data, int sz, int step) {
	//2*n раз - n = кол-во блоков
	__shared__ int sharedMem[SHARED_MEM_SIZE+SHARED_MEM_SIZE/31+32]; // + элементы из-за фейкового столбца + 32 для верности
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int offset =  blockDim.x*gridDim.x;
	int id1, baseBlockIdx;
	for (int x = idx; x < sz; x += offset) {

		baseBlockIdx = (step % 2) * BLOCK_SIZE + 2 * BLOCK_SIZE*(x / (BLOCK_SIZE));
		if (baseBlockIdx + 2 * BLOCK_SIZE > sz) {
			break;
		}
		id1 = baseBlockIdx+threadIdx.x;
		//id2 = baseBlockIdx + 2 * BLOCK_SIZE - threadIdx.x - 1;
		//  Находим начало блока -> находим конец блока -> находим противоположный элемент нашему + сдвиг от номера шага
		sharedMem[_sharedIdx(threadIdx.x)] = data[id1];
		sharedMem[_sharedIdx(threadIdx.x + BLOCK_SIZE)] = data[baseBlockIdx + threadIdx.x+BLOCK_SIZE];
		__syncthreads();

		BitonicMerge(sharedMem); 
		__syncthreads();
			   		
		data[id1] = sharedMem[_sharedIdx(threadIdx.x)];
		data[id1 + BLOCK_SIZE] = sharedMem[_sharedIdx(threadIdx.x + BLOCK_SIZE)];
		__syncthreads(); 

		}
}

int FindAddition(int sz) {
	int maxcap = 1 << 30;
	if (sz > maxcap) {
		fprintf(stderr, "ERROR: SIZE is above maximum capacity of int\n");
		return -1;
	}
	return sz % BLOCK_SIZE==0? 0 : BLOCK_SIZE-sz%BLOCK_SIZE;
}

int main() {
	int size, totalSize;
	int i;
	//scanf("%d", &size);
	fread(&size, sizeof(int), 1, stdin);
	int toAdd = FindAddition(size);
	if (toAdd == -1) {
		return 0;
	}
	totalSize = size + toAdd;
	int* arr = (int*)malloc(sizeof(int)*totalSize);
	fread(arr, sizeof(int), size, stdin);
	//srand(time(NULL));
	/*for (i = 0; i < size; ++i) {
		//scanf("%d", &arr[i]);
		arr[i] = rand();
		//fread(&arr[i], sizeof(int), 1, stdin);
	}*/
	for (i = size; i < totalSize; ++i) {
		arr[i] = INT_MAX;
	}

	int* GPUdata;
	CSC(cudaMalloc(&GPUdata, sizeof(int)*totalSize));
	CSC(cudaMemcpy(GPUdata, arr, sizeof(int)*totalSize, cudaMemcpyHostToDevice));
	EvenOddPreSort << <128, BLOCK_SIZE >> > (GPUdata, totalSize);
	//CSC(cudaMemcpy(arr, GPUdata, sizeof(int)*totalSize, cudaMemcpyDeviceToHost));
	if(totalSize > BLOCK_SIZE){
		for (int k = 0; k < 2 * (totalSize / BLOCK_SIZE); ++k) {
			BitonicBlockMerge << <128, BLOCK_SIZE>> > (GPUdata, totalSize,k);
			CSC(cudaGetLastError());
		}
	}
	CSC(cudaMemcpy(arr, GPUdata, sizeof(int)*totalSize, cudaMemcpyDeviceToHost));
	//if (!CheckSorted(arr, totalSize)) {
	//	return 1;
	//}
	CheckSorted(arr, totalSize);
	fprintf(stderr, "%d\n",size);
	if(size < 100){
		for(i = 0;i<size;++i){
			fprintf(stderr, "%d ", arr[i]);
		}
		fprintf(stderr,"\n");
	}
	if(size!=0){
		fwrite(arr, sizeof(int), size, stdout);
	}
	return 0;

}

