//#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <algorithm>
#include "mpi.h"

using namespace std;
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
void copypasta(double* src, double* dst, int bsizeX, int bsizeY, int bsizeZ) {
	for (int i = 0; i < bsizeX; ++i) {
		for (int j = 0; j < bsizeY; ++j) {
			for (int k = 0; k < bsizeZ; ++k) {
				dst[_i(i, j, k)] = src[_i(i, j, k)];
			}
		}
	}
}

void printBorder(double* data, int sz1, int sz2) {
	for (int i = 0; i < sz1; ++i) {
		for (int j = 0; j < sz2; ++j) {
			printf("%e ", data[i*sz2 + j]);
		}
		printf("\n");
	}
}*/

int main(int argc, char *argv[]) {
	int gsizeX, gsizeY, gsizeZ; //grid dimensions
	int bsizeX, bsizeY, bsizeZ; //block dimensions
	double Udown, Uup, Uleft, Uright, Ufront, Uback,U0;
	double lx, ly, lz, hx, hy, hz;
	double epsilon;
	char outfile[128];
	int ib, jb, kb;
	int numproc, id;
	int i, j, k;
	double *Udata, *Unext, *buff, *Utemp;

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

	//hx = lx / (double)(gsizeX * bsizeX);
	//hy = ly / (double)(gsizeY * bsizeY);
	//hz = lz / (double)(gsizeZ * bsizeZ);


	Udata = (double *)malloc(sizeof(double) * (bsizeX + 2)*(bsizeY + 2)*(bsizeZ + 2));
	Unext = (double *)malloc(sizeof(double) * (bsizeX+2)*(bsizeY + 2)*(bsizeZ + 2));

	buff = (double *)malloc(sizeof(double) * max(max(bsizeX*bsizeY, bsizeX*bsizeZ), bsizeY*bsizeZ));

	for (i = -1; i <= bsizeX; ++i) {
		for (j = -1; j <= bsizeY; ++j) {
			for (k = -1; k <= bsizeZ; ++k) {
				Udata[_i(i, j, k)] = U0;
			}
		}
	}

	/*if (id == 0) {
		printf("Up=%e,Down=%e,Right=%e,Left=%e,Front=%e,Back=%e\n", Uup, Udown, Uright, Uleft, Ufront, Uback);
	}
	MPI_Barrier(MPI_COMM_WORLD);*/
	if (ib == 0) {
		for (j = -1; j <= bsizeY; ++j) {
			for (k = -1; k <= bsizeZ; ++k) {
				Udata[_i(-1, j, k)] = Uleft;
				Unext[_i(-1, j, k)] = Uleft;
			}
		}
	}
	if (jb == 0) {
		for (i = -1; i <= bsizeX; ++i) {
			for (k = -1; k <= bsizeZ; ++k) {
				Udata[_i(i, -1, k)] = Ufront;
				Unext[_i(i, -1, k)] = Ufront;
			}
		}
	}
	if (kb == 0) {
		for (j = -1; j <= bsizeY; ++j) {
			for (i = -1; i <= bsizeX; ++i) {
				Udata[_i(i, j, -1)] = Udown;
				Unext[_i(i, j, -1)] = Udown;
			}
		}
	}
	if (ib + 1 == gsizeX) {
		for (j = -1; j <= bsizeY; ++j) {
			for (k = -1; k <= bsizeZ; ++k) {
				Udata[_i(bsizeX, j, k)] = Uright;
				Unext[_i(bsizeX, j, k)] = Uright;
			}
		}
	}
	if (jb + 1 == gsizeY) {
		for (i = -1; i <= bsizeX; ++i) {
			for (k = -1; k <= bsizeZ; ++k) {
				Udata[_i(i, bsizeY, k)] = Uback;
				Unext[_i(i, bsizeY, k)] = Uback;
			}
		}
	}
	if (kb + 1 == gsizeZ) {
		for (j = -1; j <= bsizeY; ++j) {
			for (i = -1; i <= bsizeX; ++i) {
				Udata[_i(i, j, bsizeZ)] = Uup;
				Unext[_i(i, j, bsizeZ)] = Uup;
			}
		}
	}

	/*
	printf("Proc: %d\n", id);
	for (k = -1; k <= bsizeZ; ++k) {
		for (j = -1; j <= bsizeY; ++j) {
			for (i = -1; i <= bsizeX; ++i) {
				if (abs(Udata[_i(i, j, k)]) > 1000) {
					Udata[_i(i, j, k)] = -1;
				}
				//printf("[%d: %d,%d,%d] = %e ", id, i, j, k, Udata[_i(i, j, k)]);
				printf("%e ", Udata[_i(i, j, k)]);
			}
			printf("\n");
		}
		printf("\n");
	}

	   

	MPI_Finalize();
	return 0;*/
	
	/*for(k=-1;k<=bsizeZ;++k){
		for (j = -1; j <= bsizeY; ++j) {
			for (i = -1; i <= bsizeX; ++i) {
				printf("%d ", _i(i, j, k));
			}
			printf("\n");
		}
		printf("\n");
	}*/

	double localMax = 0.0;
	double globalMax = -1;
	
	//int iter = 0;
	bool flag = true;
	//printf("[%d: %f, %f, %f]\n", id, hx, hy, hz);
	//cout << "Cycle\n";

	MPI_Barrier(MPI_COMM_WORLD);
	while (flag) {
		localMax = 0.0;

		MPI_Barrier(MPI_COMM_WORLD);
		// Вправо Вперёд Вверх

		if (ib + 1 < gsizeX) {
			for (j = 0; j < bsizeY; ++j) {
				for (k = 0; k < bsizeZ; ++k) {
					buff[j*bsizeZ + k] = Udata[_i(bsizeX - 1, j, k)];
				}
			}
			MPI_Send(buff, bsizeY*bsizeZ, MPI_DOUBLE, _ib(ib + 1, jb, kb), 0, MPI_COMM_WORLD);
		}

		if (ib > 0) {
			MPI_Recv(buff, bsizeY*bsizeZ, MPI_DOUBLE, _ib(ib - 1, jb, kb), 0, MPI_COMM_WORLD, &status);
			for (j = 0; j < bsizeY; ++j) {
				for (k = 0; k < bsizeZ; ++k) {
					Udata[_i(-1, j, k)] = buff[j*bsizeZ + k];
				}
			}
		}

		if (jb + 1 < gsizeY) {
			for (i = 0; i < bsizeX; ++i) {
				for (k = 0; k < bsizeZ; ++k) {
					buff[i*bsizeZ + k] = Udata[_i(i, bsizeY - 1, k)];
				}
			}
			MPI_Send(buff, bsizeX*bsizeZ, MPI_DOUBLE, _ib(ib, jb + 1, kb), 0, MPI_COMM_WORLD);
		}

		if (jb > 0) {
			MPI_Recv(buff, bsizeX*bsizeZ, MPI_DOUBLE, _ib(ib, jb - 1, kb), 0, MPI_COMM_WORLD, &status);
			for (i = 0; i < bsizeX; ++i) {
				for (k = 0; k < bsizeZ; ++k) {
					Udata[_i(i, -1, k)] = buff[i*bsizeZ + k];
				}
			}
		}

		if (kb + 1 < gsizeZ) {
			for (i = 0; i < bsizeX; ++i) {
				for (j = 0; j < bsizeY; ++j) {
					buff[i*bsizeY + j] = Udata[_i(i, j, bsizeZ - 1)];
				}
			}
			/*if (iter == 68) {
				printf("%d send to %d\n", id, _ib(ib, jb, kb + 1));
				printBorder(buff, bsizeX, bsizeY);
			}*/
			MPI_Send(buff, (bsizeY*bsizeX), MPI_DOUBLE, _ib(ib, jb, kb + 1), 0, MPI_COMM_WORLD);
		}
		
		if (kb > 0) {
			MPI_Recv(buff, (bsizeY*bsizeX), MPI_DOUBLE, _ib(ib, jb, kb-1), 0, MPI_COMM_WORLD, &status);
			/*if (iter == 68) {
				printf("%d recieved from %d\n", id, _ib(ib, jb, kb - 1));
				printBorder(buff, bsizeX, bsizeY);
			}*/
			for (i = 0; i < bsizeX; ++i) {
				for (j = 0; j < bsizeY; ++j) {
					Udata[_i(i, j, -1)] = buff[i*bsizeY + j];
				}
			}
		}
		

		//Влево Назад Вниз


		if (ib > 0) {
			for (j = 0; j < bsizeY; ++j) {
				for (k = 0; k < bsizeZ; ++k) {
					buff[j*bsizeZ + k] = Udata[_i(0, j, k)];
				}
			}
			MPI_Send(buff, bsizeY*bsizeZ, MPI_DOUBLE, _ib(ib - 1, jb, kb), 0, MPI_COMM_WORLD);
		}

		if (ib + 1 < gsizeX) {
			MPI_Recv(buff, bsizeY*bsizeZ, MPI_DOUBLE, _ib(ib + 1, jb, kb), 0, MPI_COMM_WORLD, &status);
			for (j = 0; j < bsizeY; ++j) {
				for (k = 0; k < bsizeZ; ++k) {
					Udata[_i(bsizeX, j, k)] = buff[j*bsizeZ + k];
				}
			}
		}

		if (jb > 0) {
			for (i = 0; i < bsizeX; ++i) {
				for (k = 0; k < bsizeZ; ++k) {
					buff[i*bsizeZ + k] = Udata[_i(i, 0, k)];
				}
			}
			MPI_Send(buff, bsizeX*bsizeZ, MPI_DOUBLE, _ib(ib, jb - 1, kb), 0, MPI_COMM_WORLD);
		}

		if (jb + 1 < gsizeY) {
			MPI_Recv(buff, bsizeX*bsizeZ, MPI_DOUBLE, _ib(ib, jb + 1, kb), 0, MPI_COMM_WORLD, &status);
			for (i = 0; i < bsizeX; ++i) {
				for (k = 0; k < bsizeZ; ++k) {
					Udata[_i(i, bsizeY, k)] = buff[i*bsizeZ + k];
				}
			}
		}

		if (kb > 0) {
			for (i = 0; i < bsizeX; ++i) {
				for (j = 0; j < bsizeY; ++j) {
					buff[i*bsizeY + j] = Udata[_i(i, j, 0)];
				}
			}
			/*if (iter == 68) {
				printf("%d send to %d\n", id, _ib(ib, jb, kb - 1));
				printBorder(buff, bsizeX, bsizeY);
			}*/
			MPI_Send(buff, (bsizeY*bsizeX), MPI_DOUBLE, _ib(ib, jb, kb - 1), 0, MPI_COMM_WORLD);
		}

		if (kb + 1 < gsizeZ) {
			MPI_Recv(buff, (bsizeY*bsizeX), MPI_DOUBLE, _ib(ib, jb, kb + 1), 0, MPI_COMM_WORLD, &status);
			/*if (iter == 68) {
				printf("%d recieved from %d\n", id, _ib(ib, jb, kb + 1));
				printBorder(buff, bsizeX, bsizeY);
			}*/
			for (i = 0; i < bsizeX; ++i) {
				for (j = 0; j < bsizeY; ++j) {
					Udata[_i(i, j, bsizeZ)] = buff[i*bsizeY + j];
				}
			}
		}


		MPI_Barrier(MPI_COMM_WORLD);
		
		for (i = 0; i < bsizeX; i++) {
			for (j = 0; j < bsizeY; j++) {
				for (k = 0; k < bsizeZ; k++) {
					Unext[_i(i, j, k)] = 
						0.5*(
						((Udata[_i(i + 1, j, k)] + Udata[_i(i - 1, j, k)]) / (hx * hx)) +
						((Udata[_i(i, j + 1, k)] + Udata[_i(i, j - 1, k)]) / (hy * hy)) +
						((Udata[_i(i, j, k + 1)] + Udata[_i(i, j, k - 1)]) / (hz * hz))
						)
						/
						(1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz));
					if (abs(Unext[_i(i, j, k)] - Udata[_i(i, j, k)]) > localMax) {
						localMax = abs(Unext[_i(i, j, k)] - Udata[_i(i, j, k)]);
					}
				}
			}
		}
		
		
		Utemp = Unext;
		Unext = Udata;
		Udata = Utemp;
		//copypasta(Unext, Udata, bsizeX, bsizeY, bsizeZ);


		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		if (globalMax < epsilon) {
			//printf("Convergation achieved!GMax=%e Iteration: %d\n",globalMax,iter);
			flag = false;
		}

		//MPI_Barrier(MPI_COMM_WORLD);
		/*if (iter > 5000) {
			printf("Iterartion limit reached\n");
			flag = false;
		}*/
		//iter++;

		/*
		for (i = -1; i <= bsizeX; ++i) {
			for (j = -1; j <= bsizeY; ++j) {
				for (k = -1; k <= bsizeZ; ++k) {
					if (abs(Udata[_i(i, j, k)]) > 1000) {
						Udata[_i(i, j, k)] = -1;
					}
					printf("[%d: %d,%d,%d] = %f ", id, i, j, k, Udata[_i(i, j, k)]);
				}
				printf("\n");
			}
		}
		end = true;*/

	}
	/*printf("Proc: %d\n", id);
	for (k = 0; k < bsizeZ; ++k) {
		for (j = 0; j < bsizeY; ++j) {
			for (i = 0; i < bsizeX; ++i) {
				if (abs(Udata[_i(i, j, k)]) > 1000) {
					//Udata[_i(i, j, k)] = -1;
				}
				//printf("[%d: %d,%d,%d] = %e ", id, i, j, k, Udata[_i(i, j, k)]);
				printf("%e ", Udata[_i(i, j, k)]);
			}
			printf("\n");
		}
		printf("\n");
	}*/

	MPI_Barrier(MPI_COMM_WORLD);
	// Печать
	if (id != 0) {
		for (k = 0; k < bsizeZ; ++k) {
			for (j = 0; j < bsizeY; ++j) {
				for (i = 0; i < bsizeX; ++i) {
					buff[i] = Udata[_i(i, j, k)];
				}
				MPI_Send(buff, bsizeX, MPI_DOUBLE, 0, id, MPI_COMM_WORLD);
			}
		}
	}
	else {
		FILE* file = fopen(outfile, "w");
		for (kb = 0; kb < gsizeZ; ++kb) {
			for (k = 0; k < bsizeZ; ++k) {
				for (jb = 0; jb < gsizeY; ++jb) {
					for (j = 0; j < bsizeY; ++j) {
						for (ib = 0; ib < gsizeX; ++ib) {
							if (_ib(ib, jb, kb) == 0) {
								for (i = 0; i < bsizeX; ++i) {
									buff[i] = Udata[_i(i, j, k)];
								}
							}
							else {
								MPI_Recv(buff, bsizeX, MPI_DOUBLE, _ib(ib, jb, kb), _ib(ib, jb, kb), MPI_COMM_WORLD, &status);
							}

							for (i = 0; i < bsizeX; ++i) {
								fprintf(file, "%e ", buff[i]);
							}
							fprintf(file, "\n");


						}
					}
				}
			}
		}
		fclose(file);
	}
	MPI_Finalize();

	free(buff);
	free(Udata);
	free(Unext);

	return 0;
}


