
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#define _CRT_SECURE_NO_WARNINGS
#include <thrust/extrema.h>
#include <thrust/device_vector.h> 
#include <iostream>
#include <algorithm>
#include "mpi.h"

using namespace std;


#define CSC(call)  												\
do {															\
														\
		cudaError_t res = call;									\
		if (res != cudaSuccess) {								\
			fprintf(stderr, "ERROR in %s:%d thread %d. Message: %s\n",	\
				__FILE__, __LINE__,id, cudaGetErrorString(res));	\
			exit(0);											\
		}														\
																\
} while(0)


// В блоке по координатам
#define _i(i, j, k) ( ((k)+1)*(bsizeX+2)*(bsizeY+2) + ((j)+1)*(bsizeX+2) + (i) + 1 )
// Координаты в блоке по индексу
//#define _ix(id) (((id) % (bsizeX + 2)) - 1)
//#define _iy(id) ( ( ( (id) % ( (bsizeX + 2) * (bsizeY+2)))/ (bsizeX+2) ) - 1)
//#define _iz(id) (((id)/((bsizeX+2)*(bsizeY+2))) - 1)

// По блокам покоординатно
#define _ib(i, j, k) ((k)*gsizeX*gsizeY + (j)*gsizeX + (i))
// Блок по id
#define _ibx(id) (((id) % (gsizeX * gsizeY)) % gsizeX)
#define _iby(id) (((id) % (gsizeX * gsizeY)) / gsizeX)
#define _ibz(id) ((id) / (gsizeX * gsizeY))

/*
void printBorder(double* data, int sz1, int sz2) {
	for (int i = 0; i < sz1; ++i) {
		for (int j = 0; j < sz2; ++j) {
			printf("%e ", data[i*sz2 + j]);
		}
		printf("\n");
	}
}*/

#define LEFT 1
#define RIGHT 2
#define FRONT 4
#define BACK 8
#define DOWN 16
#define UP 32
#define SEND 0
#define RECIEVE 1

__constant__ double GPUepsilon;

struct comparator {
	__device__ bool operator()(double a,double b){
		return fabs(a - b) < GPUepsilon;
	}
};

__global__ void CopyBorder(double* Udata, double* border, int bsizeX, int bsizeY, int bsizeZ, int borderIdx,int copyDirection) {

	// copyDirection - 0=send,1=recieve
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x*gridDim.x;
	int offsety = blockDim.y*gridDim.y;
	if (borderIdx == LEFT) {
		if (copyDirection == SEND) {
			for (int j = idy; j < bsizeY; j += offsety) {
				for (int k = idx; k < bsizeZ; k += offsetx) {
					border[j*bsizeZ + k] = Udata[_i(0, j, k)];
				}
			}
		}
		if (copyDirection == RECIEVE) {
			for (int j = idy; j < bsizeY; j += offsety) {
				for (int k = idx; k < bsizeZ; k += offsetx) {
					Udata[_i(-1, j, k)]= border[j*bsizeZ + k] ;
				}
			}
		}
		return;
	}
	if (borderIdx == RIGHT) {
		if (copyDirection == SEND) {
			for (int j = idy; j < bsizeY; j += offsety) {
				for (int k = idx; k < bsizeZ; k += offsetx) {
					border[j*bsizeZ + k] = Udata[_i(bsizeX-1, j, k)];
				}
			}
		}
		if (copyDirection == RECIEVE) {
			for (int j = idy; j < bsizeY; j += offsety) {
				for (int k = idx; k < bsizeZ; k += offsetx) {
					Udata[_i(bsizeX, j, k)] = border[j*bsizeZ + k];
				}
			}
		}
		return;
	}
	if (borderIdx == FRONT) {  // Пришлось поменять формат границы для сохранения объединения запросов и в границе, и в дате
		if (copyDirection == SEND) {
			for (int k = idy; k < bsizeZ; k += offsety) {
				for (int i = idx; i < bsizeX; i += offsetx) {
					border[k*bsizeX + i] = Udata[_i(i, 0, k)];
				}
			}
		}
		if (copyDirection == RECIEVE) {
			for (int k = idy; k < bsizeZ; k += offsety) {
				for (int i = idx; i < bsizeX; i += offsetx) {
					Udata[_i(i, -1, k)] = border[k*bsizeX+i];
				}
			}
		}
		return;
	}
	if (borderIdx == BACK) {
		if (copyDirection == SEND) {
			for (int k = idy; k < bsizeZ; k += offsety) {
				for (int i = idx; i < bsizeX; i += offsetx) {
					border[k*bsizeX + i] = Udata[_i(i, bsizeY-1, k)];
				}
			}
		}
		if (copyDirection == RECIEVE) {
			for (int k = idy; k < bsizeZ; k += offsety) {
				for (int i = idx; i < bsizeX; i += offsetx) {
					Udata[_i(i, bsizeY, k)] = border[k*bsizeX+i];
				}
			}
		}
		return;
	}
	if (borderIdx == DOWN) {
		if (copyDirection == SEND) {
			for (int j = idy; j < bsizeY; j += offsety) {
				for (int i = idx; i < bsizeX; i += offsetx) {
					border[j*bsizeX + i] = Udata[_i(i, j, 0)];
				}
			}
		}
		if (copyDirection == RECIEVE) {
			for (int j = idy; j < bsizeY; j += offsety) {
				for (int i = idx; i < bsizeX; i += offsetx) {
					Udata[_i(i, j, -1)] = border[j*bsizeX + i];
				}
			}
		}
		return;
	}
	if (borderIdx == UP) {
		if (copyDirection == SEND) {
			for (int j = idy; j < bsizeY; j += offsety) {
				for (int i = idx; i < bsizeX; i += offsetx) {
					border[j*bsizeX + i] = Udata[_i(i, j, bsizeZ-1)];
				}
			}
		}
		if (copyDirection==RECIEVE) {
			for (int j = idy; j < bsizeY; j += offsety) {
				for (int i = idx; i < bsizeX; i += offsetx) {
					Udata[_i(i, j, bsizeZ)] = border[j*bsizeX + i];
				}
			}
		}
		return;
	}
}

__global__ void InternalCopyBorders(double* Udata, double* Unext, int bsizeX, int bsizeY, int bsizeZ, int bordersToCopy) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int offsetx = blockDim.x*gridDim.x;
	int offsety = blockDim.y*gridDim.y;
	int i, j, k;
	if (bordersToCopy&LEFT) {
		for (j = idx; j < bsizeY; j += offsetx) {
			for (k = idy; k < bsizeZ; k += offsety) {
				Unext[_i(-1, j, k)] = Udata[_i(-1, j, k)];
			}
		}
	}
	if (bordersToCopy&RIGHT) {
		for (j = idx; j < bsizeY; j += offsetx) {
			for (k = idy; k < bsizeZ; k += offsety) {
				Unext[_i(bsizeX, j, k)] = Udata[_i(bsizeX, j, k)];
			}
		}
	}
	if (bordersToCopy&FRONT) {
		for (i = idx; i < bsizeX; i += offsetx) {
			for (k = idy; k < bsizeZ; k += offsety) {
				Unext[_i(i, -1, k)] = Udata[_i(i, -1, k)];
			}
		}
	}
	if (bordersToCopy&BACK) {
		for (i = idx; i < bsizeX; i += offsetx) {
			for (k = idy; k < bsizeZ; k += offsety) {
				Unext[_i(i, bsizeY, k)] = Udata[_i(i, bsizeY, k)];
			}
		}
	}
	if (bordersToCopy&DOWN) {
		for (i = idx; i < bsizeX; i += offsetx) {
			for (j = idy; j < bsizeY; j += offsety) {
				Unext[_i(i, j, -1)] = Udata[_i(i, j, -1)];
			}
		}
	}
	if (bordersToCopy&UP) {
		for (i = idx; i < bsizeX; i += offsetx) {
			for (j = idy; j < bsizeY; j += offsety) {
				Unext[_i(i, j, bsizeZ)] = Udata[_i(i, j, bsizeZ)];
			}
		}
	}

}


__global__ void MainKernel(double* Udata, double* Unext, int bsizeX, int bsizeY, int bsizeZ, double hx,double hy,double hz) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int idz = blockIdx.z * blockDim.z + threadIdx.z;
	int offsetx = blockDim.x*gridDim.x;
	int offsety = blockDim.y*gridDim.y; 
	int offsetz = blockDim.z*gridDim.z; 
	for (int i = idx; i < bsizeX; i+=offsetx) {
		for (int j = idy; j < bsizeY; j+=offsety) {
			for (int k = idz; k < bsizeZ; k+=offsetz) {
				Unext[_i(i, j, k)] =
					0.5*(
					((Udata[_i(i + 1, j, k)] + Udata[_i(i - 1, j, k)]) / (hx * hx)) +
						((Udata[_i(i, j + 1, k)] + Udata[_i(i, j - 1, k)]) / (hy * hy)) +
						((Udata[_i(i, j, k + 1)] + Udata[_i(i, j, k - 1)]) / (hz * hz))
						)
					/
					(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
				//if (abs(Unext[_i(i, j, k)] - Udata[_i(i, j, k)]) > localMax) {
				//	localMax = abs(Unext[_i(i, j, k)] - Udata[_i(i, j, k)]);
				//}
			}
		}
	}
}


int main(int argc, char *argv[]) {
	int gsizeX, gsizeY, gsizeZ; //grid dimensions
	int bsizeX, bsizeY, bsizeZ; //block dimensions
	double Udown, Uup, Uleft, Uright, Ufront, Uback, U0;
	double lx, ly, lz, hx, hy, hz;
	double epsilon;
	char outfile[128];
	int ib, jb, kb;
	int numproc, id;
	int i, j, k;
	double *Udata, *buff, *Utemp;
	int deviceCount;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	MPI_Barrier(MPI_COMM_WORLD);

	if (id == 0) {					// Инициализация параметров расчета
								// Размер блока по одному измерению
								// Размер сетки блоков (процессов) по одному измерению	
		cin >> gsizeX >> gsizeY >> gsizeZ;
		cin >> bsizeX >> bsizeY >> bsizeZ;
		cin >> outfile;
		cin >> epsilon;
		cin >> lx >> ly >> lz;
		cin >> Udown >> Uup >> Uleft >> Uright >> Ufront >> Uback;
		cin >> U0;
		//cout << "That's all" << endl;

		hx = lx / (double)(gsizeX * bsizeX);
		hy = ly / (double)(gsizeY * bsizeY);
		hz = lz / (double)(gsizeZ * bsizeZ);


	}
	MPI_Bcast(outfile, 128, MPI_CHAR, 0, MPI_COMM_WORLD);

	MPI_Bcast(&bsizeX, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bsizeY, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&bsizeZ, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Bcast(&gsizeX, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&gsizeY, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&gsizeZ, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Bcast(&hx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&hy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&hz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Bcast(&Udown, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Uup, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Uleft, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Uright, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Ufront, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Uback, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&U0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	ib = _ibx(id);		// Переход к 2-мерной индексации процессов 
	jb = _iby(id);
	kb = _ibz(id);

	Udata = (double *)malloc(sizeof(double) * (bsizeX + 2)*(bsizeY + 2)*(bsizeZ + 2));

	buff = (double *)malloc(sizeof(double) * max(max(bsizeX*bsizeY, bsizeX*bsizeZ), bsizeY*bsizeZ));

	for (i = -1; i <= bsizeX; ++i) {
		for (j = -1; j <= bsizeY; ++j) {
			for (k = -1; k <= bsizeZ; ++k) {
				Udata[_i(i, j, k)] = U0;
			}
		}
	}

	int fillingTemplate = 0;
	if (ib == 0) {
		for (j = -1; j <= bsizeY; ++j) {
			for (k = -1; k <= bsizeZ; ++k) {
				Udata[_i(-1, j, k)] = Uleft;
			}
		}
	}
	else {
		fillingTemplate |= LEFT;
	}
	if (jb == 0) {
		for (i = -1; i <= bsizeX; ++i) {
			for (k = -1; k <= bsizeZ; ++k) {
				Udata[_i(i, -1, k)] = Ufront;
			}
		}
	}
	else {
		fillingTemplate |= FRONT;
	}
	if (kb == 0) {
		for (j = -1; j <= bsizeY; ++j) {
			for (i = -1; i <= bsizeX; ++i) {
				Udata[_i(i, j, -1)] = Udown;
			}
		}
	}
	else {
		fillingTemplate |= DOWN;
	}
	if (ib + 1 == gsizeX) {
		for (j = -1; j <= bsizeY; ++j) {
			for (k = -1; k <= bsizeZ; ++k) {
				Udata[_i(bsizeX, j, k)] = Uright;
			}
		}
	}
	else {
		fillingTemplate |= RIGHT;
	}
	if (jb + 1 == gsizeY) {
		for (i = -1; i <= bsizeX; ++i) {
			for (k = -1; k <= bsizeZ; ++k) {
				Udata[_i(i, bsizeY, k)] = Uback;
			}
		}
	}
	else {
		fillingTemplate |= BACK;
	}
	if (kb + 1 == gsizeZ) {
		for (j = -1; j <= bsizeY; ++j) {
			for (i = -1; i <= bsizeX; ++i) {
				Udata[_i(i, j, bsizeZ)] = Uup;
			}
		}
	}
	else {
		fillingTemplate |= UP;
	}



	//double localMax = 0.0;
	//double globalMax = -1;
	int convergence = 0;
	int globalConvergence = -1;


	//printf("%d: %d", id, fillingTemplate);
	//return 0;
	CSC(cudaGetDeviceCount(&deviceCount));
	//printf("Devices: %d\n", deviceCount);
	if (deviceCount != 1) {
			
		CSC(cudaSetDevice(id & deviceCount));
	}
	double* GPUdata, *GPUnext, *GPUbuff;
	thrust::device_ptr<double> thrGPUdata, thrGPUnext, thrGPUtemp;
	comparator comp;
	CSC(cudaMalloc(&GPUdata, sizeof(double)*(bsizeX + 2)*(bsizeY + 2)*(bsizeZ + 2)));
	CSC(cudaMalloc(&GPUnext, sizeof(double)*(bsizeX + 2)*(bsizeY + 2)*(bsizeZ + 2)));
	CSC(cudaMalloc(&GPUbuff, sizeof(double)*max(max(bsizeX*bsizeY, bsizeX*bsizeZ), bsizeY*bsizeZ)));
	thrGPUdata = thrust::device_pointer_cast(GPUdata);
	thrGPUnext = thrust::device_pointer_cast(GPUnext);
	
	CSC(cudaMemcpy(GPUdata, Udata, sizeof(double)*(bsizeX + 2)*(bsizeY + 2)*(bsizeZ + 2), cudaMemcpyHostToDevice));
	CSC(cudaMemcpy(GPUnext, Udata, sizeof(double)*(bsizeX + 2)*(bsizeY + 2)*(bsizeZ + 2), cudaMemcpyHostToDevice));
	CSC(cudaMemcpyToSymbol(GPUepsilon, &epsilon, sizeof(double), 0, cudaMemcpyHostToDevice));
	
	

	//int iter = 0;
	bool flag = true;
	//printf("[%d: %f, %f, %f]\n", id, hx, hy, hz);
	//cout << "Cycle\n";
	//
	MPI_Barrier(MPI_COMM_WORLD);
	while (flag) {
		//localMax = 0.0;
		convergence = 0;
		MPI_Barrier(MPI_COMM_WORLD);
		// Вправо Вперёд Вверх
			   		 	  	  	   	
		if (ib + 1 < gsizeX) {
			CopyBorder<<<32,32>>>(GPUdata, GPUbuff, bsizeX, bsizeY, bsizeZ, RIGHT, SEND);
			CSC(cudaMemcpy(buff, GPUbuff, sizeof(double)*bsizeY*bsizeZ, cudaMemcpyDeviceToHost));
			MPI_Send(buff, bsizeY*bsizeZ, MPI_DOUBLE, _ib(ib + 1, jb, kb), 0, MPI_COMM_WORLD);
		}

		if (ib > 0) {
			MPI_Recv(buff, bsizeY*bsizeZ, MPI_DOUBLE, _ib(ib - 1, jb, kb), 0, MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(GPUbuff, buff, sizeof(double)*bsizeY*bsizeZ, cudaMemcpyHostToDevice));
			CopyBorder << <32, 32 >> > (GPUdata, GPUbuff, bsizeX, bsizeY, bsizeZ, LEFT, RECIEVE);
		}

		if (jb + 1 < gsizeY) {
			CopyBorder << <32, 32 >> > (GPUdata, GPUbuff, bsizeX, bsizeY, bsizeZ, BACK, SEND);
			CSC(cudaMemcpy(buff, GPUbuff, sizeof(double)*bsizeX*bsizeZ, cudaMemcpyDeviceToHost));
			MPI_Send(buff, bsizeX*bsizeZ, MPI_DOUBLE, _ib(ib, jb + 1, kb), 0, MPI_COMM_WORLD);
		}

		if (jb > 0) {
			MPI_Recv(buff, bsizeX*bsizeZ, MPI_DOUBLE, _ib(ib, jb - 1, kb), 0, MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(GPUbuff, buff, sizeof(double)*bsizeX*bsizeZ, cudaMemcpyHostToDevice));
			CopyBorder << <32, 32 >> > (GPUdata, GPUbuff, bsizeX, bsizeY, bsizeZ, FRONT, RECIEVE);
		}

		if (kb + 1 < gsizeZ) {
			CopyBorder << <32, 32 >> > (GPUdata, GPUbuff, bsizeX, bsizeY, bsizeZ, UP, SEND);
			CSC(cudaMemcpy(buff, GPUbuff, sizeof(double)*bsizeY*bsizeX, cudaMemcpyDeviceToHost));

			MPI_Send(buff, (bsizeY*bsizeX), MPI_DOUBLE, _ib(ib, jb, kb + 1), 0, MPI_COMM_WORLD);
		}

		if (kb > 0) {
			MPI_Recv(buff, (bsizeY*bsizeX), MPI_DOUBLE, _ib(ib, jb, kb - 1), 0, MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(GPUbuff, buff, sizeof(double)*bsizeY*bsizeX, cudaMemcpyHostToDevice));
			CopyBorder << <32, 32 >> > (GPUdata, GPUbuff, bsizeX, bsizeY, bsizeZ, DOWN, RECIEVE);
		}


		//Влево Назад Вниз


		if (ib > 0) {
			CopyBorder << <32, 32 >> > (GPUdata, GPUbuff, bsizeX, bsizeY, bsizeZ, LEFT, SEND);
			CSC(cudaMemcpy(buff, GPUbuff, sizeof(double)*bsizeY*bsizeZ, cudaMemcpyDeviceToHost));
			MPI_Send(buff, bsizeY*bsizeZ, MPI_DOUBLE, _ib(ib - 1, jb, kb), 0, MPI_COMM_WORLD);
		}

		if (ib + 1 < gsizeX) {
			MPI_Recv(buff, bsizeY*bsizeZ, MPI_DOUBLE, _ib(ib + 1, jb, kb), 0, MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(GPUbuff, buff, sizeof(double)*bsizeY*bsizeZ, cudaMemcpyHostToDevice));
			CopyBorder << <32, 32 >> > (GPUdata, GPUbuff, bsizeX, bsizeY, bsizeZ, RIGHT, RECIEVE);
		}

		if (jb > 0) {
			CopyBorder << <32, 32 >> > (GPUdata, GPUbuff, bsizeX, bsizeY, bsizeZ, FRONT, SEND);
			CSC(cudaMemcpy(buff, GPUbuff, sizeof(double)*bsizeX*bsizeZ, cudaMemcpyDeviceToHost));
			MPI_Send(buff, bsizeX*bsizeZ, MPI_DOUBLE, _ib(ib, jb - 1, kb), 0, MPI_COMM_WORLD);
		}

		if (jb + 1 < gsizeY) {
			MPI_Recv(buff, bsizeX*bsizeZ, MPI_DOUBLE, _ib(ib, jb + 1, kb), 0, MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(GPUbuff, buff, sizeof(double)*bsizeX*bsizeZ, cudaMemcpyHostToDevice));
			CopyBorder << <32, 32 >> > (GPUdata, GPUbuff, bsizeX, bsizeY, bsizeZ, BACK, RECIEVE);
		}

		if (kb > 0) {
			CopyBorder << <32, 32 >> > (GPUdata, GPUbuff, bsizeX, bsizeY, bsizeZ, DOWN, SEND);
			CSC(cudaMemcpy(buff, GPUbuff, sizeof(double)*bsizeY*bsizeX, cudaMemcpyDeviceToHost));
			MPI_Send(buff, (bsizeY*bsizeX), MPI_DOUBLE, _ib(ib, jb, kb - 1), 0, MPI_COMM_WORLD);
		}

		if (kb + 1 < gsizeZ) {
			MPI_Recv(buff, (bsizeY*bsizeX), MPI_DOUBLE, _ib(ib, jb, kb + 1), 0, MPI_COMM_WORLD, &status);
			CSC(cudaMemcpy(GPUbuff, buff, sizeof(double)*bsizeY*bsizeX, cudaMemcpyHostToDevice));
			CopyBorder << <32, 32 >> > (GPUdata, GPUbuff, bsizeX, bsizeY, bsizeZ, UP, RECIEVE);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		
		MainKernel<<<32,32,32>>>(GPUdata, GPUnext, bsizeX, bsizeY, bsizeZ, hx, hy, hz);

		InternalCopyBorders<<<32,32>>>(GPUdata, GPUnext, bsizeX, bsizeY, bsizeZ, fillingTemplate);
		//AllReduce находит теперь минимум. Если хоть один не сошелся - 0 при сверке.
		if (thrust::equal(thrGPUdata, thrGPUdata + (bsizeX + 2)*(bsizeY + 2)*(bsizeZ + 2), thrGPUnext, comp)) {
			convergence = 1;
		}
		else {
			convergence = 0;
		}
		
		Utemp = GPUnext;
		GPUnext = GPUdata;
		GPUdata = Utemp;

		thrGPUtemp = thrGPUdata;
		thrGPUdata = thrGPUnext;
		thrGPUnext = thrGPUtemp;

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(&convergence, &globalConvergence, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		if (globalConvergence==1) {
			//printf("Convergation achieved!Iteration: %d\n",iter);
			flag = false;
		}

	}
	
	CSC(cudaMemcpy(Udata, GPUdata, sizeof(double)*(bsizeX + 2)*(bsizeY + 2)*(bsizeZ + 2), cudaMemcpyDeviceToHost));
	CSC(cudaFree(GPUdata));	
	CSC(cudaFree(GPUnext));
	CSC(cudaFree(GPUbuff));
	
	MPI_Barrier(MPI_COMM_WORLD);
	// Печать

	int charPerDouble = 14;
	char* charBuff = (char*)malloc(sizeof(char)*bsizeX*bsizeY*bsizeZ*charPerDouble);
	memset(charBuff, ' ', bsizeX*bsizeY*bsizeZ*charPerDouble*sizeof(char));
	for (k = 0; k < bsizeZ; ++k) {
		for (j = 0; j < bsizeY; ++j) {
			for (i = 0; i < bsizeX; ++i) {
				sprintf(charBuff + (k*bsizeY*bsizeX + j * bsizeX + i)*charPerDouble, "%e", Udata[_i(i, j, k)]);
			}
			//if (ib + 1 == gsizeX) {
				charBuff[(k*bsizeY*bsizeX + (j + 1)*bsizeX)*charPerDouble - 1] = '\n';
			//}
		}
	}
	for (i = 0; i < bsizeX*bsizeY*bsizeZ*charPerDouble; ++i) {
		if (charBuff[i] == '\0') {
			charBuff[i] = ' ';
		}
	}
	
	//printf("ID: %d\n", id);
	//printf("%.*s\n",charPerDouble*bsizeX*bsizeY*bsizeZ, charBuff);

	MPI_Datatype newHindexedType;
	int blockCount = bsizeY * bsizeZ;
	int* blockLengths = (int*)malloc(sizeof(int)*bsizeY*bsizeZ);
	MPI_Aint* blockDisplacements = (MPI_Aint*)malloc(sizeof(MPI_Aint)*bsizeY*bsizeZ);
	for (k = 0; k < bsizeZ; ++k) {
		for (j = 0; j < bsizeY; ++j) {
			blockLengths[k*bsizeY+j] = bsizeX * charPerDouble; 
			blockDisplacements[k*bsizeY + j] = (j * gsizeX + k * gsizeX*bsizeY*gsizeY)*charPerDouble*bsizeX;
			//printf("%d ", blockDisplacements[k*bsizeY + j]);
		}
		//printf("\n");
	}
	MPI_Type_create_hindexed(blockCount, blockLengths, blockDisplacements, MPI_CHAR, &newHindexedType);
	MPI_Type_commit(&newHindexedType);
	
	MPI_File fl;
	//MPI_File_delete(outfile, MPI_INFO_NULL);
	MPI_File_open(MPI_COMM_WORLD, outfile, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fl);
	int fileOffset = (ib + jb * gsizeX * bsizeY + kb * gsizeX*gsizeY*bsizeY*bsizeZ)*bsizeX*charPerDouble;
	MPI_File_set_view(fl, fileOffset, MPI_CHAR, newHindexedType, "native", MPI_INFO_NULL);
	MPI_File_write_all(fl, charBuff, bsizeX*bsizeY*bsizeZ*charPerDouble, MPI_CHAR, &status);
	MPI_File_close(&fl);

	MPI_Type_free(&newHindexedType);
	MPI_Barrier(MPI_COMM_WORLD);
	
	MPI_Finalize();

	free(buff);
	free(Udata);

	return 0;
}
