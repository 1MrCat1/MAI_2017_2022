
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

void printim(uchar4* data,int w,int h);

__host__ __device__ uchar4 float4ToUchar4(float4 a) {
	return make_uchar4(a.x, a.y, a.z, a.w);
}

__host__ __device__ float4 uchar4ToFloat4(uchar4 a) {
	return make_float4(a.x, a.y, a.z, a.w);
}

texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void kernel(float4 *out, int w, int h, int resW, int resH) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	int multiplier = (w / resW) * (h / resH);
	uchar4 p;
	for (y = idy; y < h; y += offsety)
		for (x = idx; x < w; x += offsetx) {
			p = tex2D(tex, x, y);
			atomicAdd(&out[y / (h / resH) * resW + x / (w / resW)].x, (float)p.x / multiplier);
			atomicAdd(&out[y / (h / resH) * resW + x / (w / resW)].y, (float)p.y / multiplier);
			atomicAdd(&out[y / (h / resH) * resW + x / (w / resW)].z, (float)p.z / multiplier);
		}
}

int main() {
	int w, h;
	int resW, resH;
	char filename[256];
	char outputfilename[256];
	scanf("%s", filename);
	scanf("%s", outputfilename);
	FILE *fl = fopen(filename, "rb");
	fread(&w, sizeof(int), 1, fl);
	fread(&h, sizeof(int), 1, fl);
	uchar4 *udata = (uchar4 *)malloc(sizeof(uchar4) * w * h);
	fread(udata, sizeof(uchar4), w * h, fl);
	fclose(fl);
	scanf("%d %d", &resW, &resH);
	/*float4* fdata = (float4*)malloc(sizeof(float4)*w*h);
	for (int i = 0; i < w*h; ++i) {
		fdata[i] = uchar4ToFloat4(udata[i]);
	}*/
	
	cudaArray *arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	cudaError_t err = cudaMallocArray(&arr, &ch, w, h);
	if(err !=cudaSuccess){
		fprintf(stderr,"W=%d H=%d\n",w,h);
		CSC(err);
	}
	CSC(cudaMemcpyToArray(arr, 0, 0, udata, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

	tex.addressMode[0] = cudaAddressModeClamp;	
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;		
	tex.normalized = false;						
	
	CSC(cudaBindTextureToArray(tex, arr, ch));

	free(udata);
	//free(fdata);
	float4* fdata = (float4*)malloc(sizeof(float4)*resW*resH);
	for (int i = 0; i < resW*resH; ++i) {
		fdata[i] = make_float4(0, 0, 0, 0);
	}
	float4 *outImage;
	CSC(cudaMalloc(&outImage, sizeof(float4) * resW * resH));
	CSC(cudaMemcpy(outImage, fdata, sizeof(float4)*resW*resH, cudaMemcpyHostToDevice));
	kernel <<<dim3(16, 16), dim3(16, 16)>>> (outImage, w, h,resW,resH);
	CSC(cudaGetLastError());

	CSC(cudaMemcpy(fdata, outImage, sizeof(float4) * resW * resH, cudaMemcpyDeviceToHost));

	CSC(cudaUnbindTexture(tex));
	CSC(cudaFreeArray(arr));
	CSC(cudaFree(outImage));
	
	udata = (uchar4*)malloc(sizeof(uchar4)*resW*resH);
	for (int i = 0; i < resW*resH; ++i) {
		udata[i] = float4ToUchar4(fdata[i]);
	}
	//printim(udata, resW, resH);
	fl = fopen(outputfilename, "wb");
	fwrite(&resW, sizeof(int), 1, fl);
	fwrite(&resH, sizeof(int), 1, fl);
	fwrite(udata, sizeof(uchar4), resW * resH, fl);
	fclose(fl);

	free(udata);
	free(fdata);
	return 0;
}


void printim(uchar4* data,int w,int h) {
	printf("%d %d\n", w, h);
	if (!data) {
		printf("Empty image");
		return;
	}
	for (size_t i = 0; i < h; ++i) {
		for (size_t j = 0; j < w; ++j) {
			printf("%0X%0X%0X%0X ", data[i*w+j].x, data[i*w + j].y, data[i*w + j].z, data[i*w + j].w);
		}
		printf("\n");
	}
}