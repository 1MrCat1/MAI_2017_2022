#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <limits>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)


__host__ __device__ uchar4 float4ToUchar4(float4 a) {
	return make_uchar4(a.x, a.y, a.z, a.w);
}

__host__ __device__ float4 uchar4ToFloat4(uchar4 a) {
	return make_float4(a.x, a.y, a.z, a.w);
}

float FLOATMIN = -3.402823466e+38;

int amountOfClasses = 0;
float4 averages[32] = { 0,0,0,0 };
float covMatrix[32][3][3] = { 0 };
float covMatrixInverted[32][3][3] = { 0 };
float determinants[32] = { 0 };

__constant__ float GPUFLOATMIN = -3.402823466e+38;
__constant__ int GPUCLASSES;
__constant__ float4 GPUAVG[32];
__constant__ float GPUCOVINV[32][3][3];
__constant__ float GPUDETS[32];


__host__ __device__ float4 add(float4 a, float4 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

__host__ __device__ float4 sub(float4 a, float4 b) {
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
	return a;
}

__host__ __device__ float4 mult(float4 a, float4 b) {
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}

__host__ __device__ float4 multByNumber(float4 a, float b) {
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
	return a;
}

__host__ __device__ float f4sum(float4 a) {
	return a.x + a.y + a.z + a.w;
}

void CalcInvCovMatrix() {
	for (int k = 0; k < amountOfClasses; ++k) {
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				covMatrixInverted[k][i][j] = covMatrix[k][(i + 1) % 3][(j + 1) % 3] * covMatrix[k][(i + 2) % 3][(j + 2) % 3] -
											 covMatrix[k][(i + 1) % 3][(j + 2) % 3] * covMatrix[k][(i + 2) % 3][(j + 1) % 3];
				covMatrixInverted[k][i][j] /= determinants[k];
			}
		}
	}
}

void CalcDeterminants() {
	for (int i = 0; i < amountOfClasses; ++i) {
		for (int j = 0; j < 3; ++j) {
			determinants[i] += covMatrix[i][0][j] * covMatrix[i][1][(j + 1) % 3] * covMatrix[i][2][(j + 2) % 3];
			determinants[i] -= covMatrix[i][0][j] * covMatrix[i][1][(j + 2) % 3] * covMatrix[i][2][(j + 1) % 3];
		}
	}
}

void CalcCovMatrix(uchar4* data, int** basePoints,int w,int h) {
	float4 buffer;
	for (int i = 0; i < amountOfClasses; ++i) {
		for (int j = 1; j <= basePoints[i][0]; ++j) {
			buffer = sub( uchar4ToFloat4(data[basePoints[i][j*2]*w+ basePoints[i][j*2-1]]), averages[i]);

			covMatrix[i][0][0] += buffer.x * buffer.x;
			covMatrix[i][0][1] += buffer.x * buffer.y;
			covMatrix[i][0][2] += buffer.x * buffer.z;
			covMatrix[i][1][0] += buffer.y * buffer.x;
			covMatrix[i][1][1] += buffer.y * buffer.y;
			covMatrix[i][1][2] += buffer.y * buffer.z;
			covMatrix[i][2][0] += buffer.z * buffer.x;
			covMatrix[i][2][1] += buffer.z * buffer.y;
			covMatrix[i][2][2] += buffer.z * buffer.z;
		}
		for (int j = 0; j < 3; ++j) {
			for (int k = 0; k < 3; ++k) {
				covMatrix[i][j][k] /= basePoints[i][0] - 1;
			}
		}
	}
}

int** GetBasePoints(uchar4* udata, int w,int h) {
	int amountOfClassBasePixels;
	FILE* input = fopen("input.txt", "r");
	fscanf(input, "%d", &amountOfClasses);
	int** basePoints = (int**)malloc(sizeof(int*)*amountOfClasses);
	for (int i = 0; i < amountOfClasses; ++i) {
		fscanf(input, "%d", &amountOfClassBasePixels);
		basePoints[i] = (int*)malloc(sizeof(int)*amountOfClassBasePixels * 2 + 1);
		basePoints[i][0] = amountOfClassBasePixels;
		for (int j = 1; j <= amountOfClassBasePixels; ++j) {
			fscanf(input, "%d %d ", &basePoints[i][j*2-1], &basePoints[i][j*2]);
		}
	}
	fclose(input);
	return basePoints;
}

void GetClassAverages(uchar4* data, int** basePoints,int w,int h) {
	for (int i = 0; i < amountOfClasses; ++i) {
		for (int j = 1; j <= basePoints[i][0]; ++j) {
			averages[i] = add(averages[i], uchar4ToFloat4(data[basePoints[i][j * 2] * w + basePoints[i][j * 2-1]]));
		}
		averages[i] = multByNumber(averages[i], (float)1/basePoints[i][0]);

	}
}

__host__ float func(float4 pixel,int classN) {
	float4 f4buffer, f4buffer2;
	float fbuffer[3] = { 0 };
	float result=0;
	f4buffer = sub(pixel, averages[classN]);
	f4buffer2 = { 0 };
	for (int j = 0; j < 3; ++j) {
		fbuffer[j] += -f4buffer.x*covMatrixInverted[classN][j][0];
		fbuffer[j] += -f4buffer.y*covMatrixInverted[classN][j][1];
		fbuffer[j] += -f4buffer.z*covMatrixInverted[classN][j][2];
	}
	f4buffer2.x = fbuffer[0];
	f4buffer2.y = fbuffer[1];
	f4buffer2.z = fbuffer[2];
	result = f4sum(mult(f4buffer2, f4buffer));
	return result;
}

__host__ int Classify(uchar4 pixel) {
	float maxval = FLOATMIN;
	float fbuffer;
	int nclass=-1;
	for (int i = 0; i < amountOfClasses; ++i) {
		fbuffer = func(uchar4ToFloat4(pixel), i);
		if (fbuffer > maxval) {
			maxval = fbuffer;
			nclass = i;
		}
	}
	return nclass;
}

__host__ void CPUkernel(uchar4* data, int size) {
	for (int i = 0; i < size; ++i) {
		data[i].w = Classify(data[i]);
	}
}

__device__ float GPUfunc(float4 pixel, int classN) {
	float4 f4buffer, f4buffer2;
	float fbuffer[3] = { 0 };
	float result = 0;
	f4buffer = sub(pixel, GPUAVG[classN]);
	f4buffer2 = { 0 };
	for (int j = 0; j < 3; ++j) {
		fbuffer[j] += -f4buffer.x*GPUCOVINV[classN][j][0];
		fbuffer[j] += -f4buffer.y*GPUCOVINV[classN][j][1];
		fbuffer[j] += -f4buffer.z*GPUCOVINV[classN][j][2];
	}
	f4buffer2.x = fbuffer[0];
	f4buffer2.y = fbuffer[1];
	f4buffer2.z = fbuffer[2];
	result = f4sum(mult(f4buffer2, f4buffer));
	return result;
}

__device__ int GPUClassify(uchar4 pixel) {
	float maxval = GPUFLOATMIN;
	float fbuffer;
	int nclass = -1;
	for (int i = 0; i < GPUCLASSES; ++i) {
		fbuffer = GPUfunc(uchar4ToFloat4(pixel), i);
		if (fbuffer > maxval) {
			maxval = fbuffer;
			nclass = i;
		}
	}
	return nclass;
}

__global__ void kernel(uchar4* data, int w, int h) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	for (y = idy; y < h; y += offsety)
		for (x = idx; x < w; x += offsetx) {
			data[y*w+x].w = GPUClassify(data[y*w+x]);
		}
}



int main() {
	int w, h,basePointsN;
	char filename[256];
	char outputfilename[256];
	scanf("%s", filename);
	scanf("%s", outputfilename);

	FILE* fl = fopen(filename, "rb");
	fread(&w, sizeof(int), 1, fl);
	fread(&h, sizeof(int), 1, fl);
	uchar4 *udata = (uchar4 *)malloc(sizeof(uchar4) * w * h);
	fread(udata, sizeof(uchar4), w * h, fl);
	fclose(fl);

	scanf("%d", &amountOfClasses);
	int** basePoints = (int**)malloc(sizeof(int*)*amountOfClasses);
	for (int i = 0; i < amountOfClasses;++i) {
		scanf("%d", &basePointsN);
		basePoints[i] = (int*)malloc(sizeof(int)*(basePointsN * 2 + 1));
		basePoints[i][0] = basePointsN;
		for (int j = 1; j <= basePointsN; ++j) {
			scanf("%d %d", &basePoints[i][j * 2 - 1], &basePoints[i][j * 2]);
		}
	}
	//int** basePoints=GetBasePoints(udata,w,h);
	
	GetClassAverages(udata, basePoints, w, h);
	CalcCovMatrix(udata, basePoints, w, h);
	CalcDeterminants();
	CalcInvCovMatrix();

	CSC(cudaMemcpyToSymbol(GPUAVG,averages,sizeof(float4)*32,0,cudaMemcpyHostToDevice));
	CSC(cudaMemcpyToSymbol(GPUCLASSES, &amountOfClasses, sizeof(int), 0, cudaMemcpyHostToDevice));
	CSC(cudaMemcpyToSymbol(GPUCOVINV, covMatrixInverted, sizeof(float) * 32*3*3, 0, cudaMemcpyHostToDevice));
	CSC(cudaMemcpyToSymbol(GPUDETS, determinants, sizeof(float) * 32, 0, cudaMemcpyHostToDevice));

	uchar4* GPUdata;
	CSC(cudaMalloc(&GPUdata, sizeof(uchar4)*w*h));
	CSC(cudaMemcpy(GPUdata, udata, sizeof(uchar4)*w*h, cudaMemcpyHostToDevice));
	kernel << <dim3(16, 16), dim3(16, 16) >> > (GPUdata, w, h);
	CSC(cudaMemcpy(udata, GPUdata, sizeof(uchar4)*w*h, cudaMemcpyDeviceToHost));


	fl = fopen(outputfilename, "wb");
	fwrite(&w, sizeof(int), 1, fl);
	fwrite(&h, sizeof(int), 1, fl);
	fwrite(udata, sizeof(uchar4), w*h, fl);
	fclose(fl);


	//CPUkernel(udata, w*h);
	for (int i = 0; i < amountOfClasses; ++i) {
		free(basePoints[i]);
	}
	free(basePoints);
	free(udata);
	CSC(cudaFree(GPUdata));
	return 0;
}
