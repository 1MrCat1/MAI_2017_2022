
#include <iostream>
#include <algorithm>
#include "mpi.h"
#include <time.h>
#include <omp.h>
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

int main(int argc, char *argv[]) {
	int gsizeX, gsizeY, gsizeZ; //grid dimensions
	int bsizeX, bsizeY, bsizeZ; //block dimensions
	double Udown, Uup, Uleft, Uright, Ufront, Uback, U0;
	double lx, ly, lz, hx, hy, hz;
	double epsilon;
	char outfile[256];
	int ib, jb, kb;
	int numproc, id;
	int i, j, k;
	double *Udata, *Unext, *Utemp;

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
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Bcast(outfile, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
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
	Unext = (double *)malloc(sizeof(double) * (bsizeX + 2)*(bsizeY + 2)*(bsizeZ + 2));

	for (i = -1; i <= bsizeX; ++i) {
		for (j = -1; j <= bsizeY; ++j) {
			for (k = -1; k <= bsizeZ; ++k) {
				Udata[_i(i, j, k)] = U0;// _i(i, j, k);
			}
		}
	}
	/*if (id == 0) {
		printBorderIndexes(Udata, bsizeX, bsizeY, bsizeZ, LEFT | RIGHT | FRONT | BACK | DOWN | UP, id);
	}*/


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
	

	MPI_Datatype X_BORDER;
	int blockCountX = (bsizeY + 2) * (bsizeZ + 2);
	int* blockLengthsX = (int*)malloc(sizeof(int)*(bsizeY + 2)*(bsizeZ + 2));
	MPI_Aint* blockDisplacementsX = (MPI_Aint*)malloc(sizeof(MPI_Aint)*(bsizeY + 2)*(bsizeZ + 2));
	//cout << "X_BORDER\n";
	for (j = 0; j <= bsizeY + 1; ++j) {
		for (k = 0; k <= bsizeZ + 1; ++k) {
			blockLengthsX[j*(bsizeZ + 2) + k] = 1;
			blockDisplacementsX[j*(bsizeZ + 2) + k] = (j * (bsizeX + 2) + k * (bsizeX + 2)*(bsizeY + 2))*sizeof(double);
			//printf("%d ", blockDisplacementsX[j*(bsizeZ+2) + k]);
		}
		//printf("\n");
	}
	MPI_Type_create_hindexed(blockCountX, blockLengthsX, blockDisplacementsX, MPI_DOUBLE, &X_BORDER);
	MPI_Type_commit(&X_BORDER);
	free(blockLengthsX);
	free(blockDisplacementsX);
	
	MPI_Datatype Y_BORDER;
	int blockCountY = (bsizeX + 2) * (bsizeZ + 2);
	int* blockLengthsY = (int*)malloc(sizeof(int)*(bsizeX + 2)*(bsizeZ + 2));
	MPI_Aint* blockDisplacementsY = (MPI_Aint*)malloc(sizeof(MPI_Aint)*(bsizeX + 2)*(bsizeZ + 2));
	//cout << "Y_BORDER\n";
	for (j = 0; j <= bsizeX + 1; ++j) {
		for (k = 0; k <= bsizeZ + 1; ++k) {
			blockLengthsY[j*(bsizeZ + 2) + k] = 1;
			blockDisplacementsY[j*(bsizeZ + 2) + k] = (j + k * (bsizeX + 2)*(bsizeY + 2)) *sizeof(double);
			//printf("%d ", blockDisplacementsY[j*(bsizeZ+2) + k]);
		}
		//printf("\n");
	}
	MPI_Type_create_hindexed(blockCountY, blockLengthsY, blockDisplacementsY, MPI_DOUBLE, &Y_BORDER);
	MPI_Type_commit(&Y_BORDER);
	free(blockLengthsY);
	free(blockDisplacementsY);

	MPI_Datatype Z_BORDER;
	int blockCountZ = (bsizeY + 2) * (bsizeX + 2);
	int* blockLengthsZ = (int*)malloc(sizeof(int)*(bsizeY + 2)*(bsizeX + 2));
	MPI_Aint* blockDisplacementsZ = (MPI_Aint*)malloc(sizeof(MPI_Aint)*(bsizeY + 2)*(bsizeX + 2));
	//cout << "Z_BORDER\n";
	for (j = 0; j <= bsizeY + 1; ++j) {
		for (k = 0; k <= bsizeX + 1; ++k) {
			blockLengthsZ[j*(bsizeX + 2) + k] = 1;
			blockDisplacementsZ[j*(bsizeX + 2) + k] = (j * (bsizeX + 2) + k) * sizeof(double);
			//printf("%d ", blockDisplacementsZ[j*(bsizeX+2) + k]);
		}
		//printf("\n");
	}
	MPI_Type_create_hindexed(blockCountZ, blockLengthsZ, blockDisplacementsZ, MPI_DOUBLE, &Z_BORDER);
	MPI_Type_commit(&Z_BORDER);
	free(blockLengthsZ);
	free(blockDisplacementsZ);

	MPI_Barrier(MPI_COMM_WORLD);



	double localMax = 0;
	double globalMax = 0;
	double temp=0;
	double hx2, hy2, hz2;
	hx2 = hx * hx;
	hy2 = hy * hy;
	hz2 = hz * hz;
	//int iter = 0;
	bool flag = true;
	//printf("[%d: %f, %f, %f]\n", id, hx, hy, hz);
	//cout << "Cycle\n";
	time_t start, end;
	MPI_Barrier(MPI_COMM_WORLD);
	start = clock();
	fprintf(stderr, "Starting cycle\n");
	while (flag) {
		localMax = 0.0;

		MPI_Barrier(MPI_COMM_WORLD);
		// Вправо Вперёд Вверх

		if (ib + 1 < gsizeX) {
			MPI_Send(Udata+(bsizeX), 1, X_BORDER, _ib(ib + 1, jb, kb), 0, MPI_COMM_WORLD);
		}

		if (ib > 0) {
			MPI_Recv(Udata, 1, X_BORDER, _ib(ib - 1, jb, kb), 0, MPI_COMM_WORLD, &status);
		}

		if (jb + 1 < gsizeY) {
			
			MPI_Send(Udata+(bsizeX+2)*(bsizeY), 1, Y_BORDER, _ib(ib, jb + 1, kb), 0, MPI_COMM_WORLD);
		}

		if (jb > 0) {
			MPI_Recv(Udata, 1, Y_BORDER, _ib(ib, jb - 1, kb), 0, MPI_COMM_WORLD, &status);
		}

		if (kb + 1 < gsizeZ) {
			MPI_Send(Udata+(bsizeX + 2)*(bsizeY + 2)*(bsizeZ), 1, Z_BORDER, _ib(ib, jb, kb + 1), 0, MPI_COMM_WORLD);
		}

		if (kb > 0) {
			MPI_Recv(Udata, 1, Z_BORDER, _ib(ib, jb, kb - 1), 0, MPI_COMM_WORLD, &status); 
		}

		//Влево Назад Вниз

		if (ib > 0) {
			MPI_Send(Udata+1, 1, X_BORDER, _ib(ib - 1, jb, kb), 0, MPI_COMM_WORLD);
		}

		if (ib + 1 < gsizeX) {
			MPI_Recv(Udata+(bsizeX+1), 1, X_BORDER, _ib(ib + 1, jb, kb), 0, MPI_COMM_WORLD, &status);
		}

		if (jb > 0) {

			MPI_Send(Udata+(bsizeX+2), 1, Y_BORDER, _ib(ib, jb - 1, kb), 0, MPI_COMM_WORLD);
		}

		if (jb + 1 < gsizeY) {
			MPI_Recv(Udata+(bsizeX+2)*(bsizeY+1), 1, Y_BORDER, _ib(ib, jb + 1, kb), 0, MPI_COMM_WORLD, &status);
		}

		if (kb > 0) {
			MPI_Send(Udata + (bsizeX + 2)*(bsizeY + 2), 1, Z_BORDER, _ib(ib, jb, kb - 1), 0, MPI_COMM_WORLD);
		}

		if (kb + 1 < gsizeZ) {
			MPI_Recv(Udata+(bsizeX + 2)*(bsizeY + 2)*(bsizeZ+1), 1, Z_BORDER, _ib(ib, jb, kb + 1), 0, MPI_COMM_WORLD, &status);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		
		#pragma omp parallel for private(i,j,k,temp) schedule(dynamic) reduction(max:localMax)
		for (i = 0; i < bsizeX; i++) {
			for (j = 0; j < bsizeY; j++) {
				for (k = 0; k < bsizeZ; k++) {
					Unext[_i(i, j, k)] =
						0.5*(
						((Udata[_i(i + 1, j, k)] + Udata[_i(i - 1, j, k)]) / (hx2)) +
							((Udata[_i(i, j + 1, k)] + Udata[_i(i, j - 1, k)]) / (hy2)) +
							((Udata[_i(i, j, k + 1)] + Udata[_i(i, j, k - 1)]) / (hz2))
							)
						/
						(1.0 / (hx2) + 1.0 / (hy2) + 1.0 / (hz2));
					temp = abs(Unext[_i(i, j, k)] - Udata[_i(i, j, k)]);
					if (temp > localMax) {
						localMax = temp;
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
		}
		iter++;*/

	}
	fprintf(stderr, "Cycle finished\n");
	MPI_Type_free(&X_BORDER);
	MPI_Type_free(&Y_BORDER);
	MPI_Type_free(&Z_BORDER);
	
	end = clock();
	MPI_Barrier(MPI_COMM_WORLD);
	//printf("Thread %d: time %d ms\n", id, (end - start));
	// Печать

	fprintf(stderr, "Starting output\n");
	int charPerDouble = 14;
	char* charBuff = (char*)malloc(sizeof(char)*bsizeX*bsizeY*bsizeZ*charPerDouble);
	memset(charBuff, ' ', bsizeX*bsizeY*bsizeZ*charPerDouble * sizeof(char));
	
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

	fprintf(stderr, "Creating output datatype\n");
	MPI_Datatype newHindexedType;
	int blockCount = bsizeY * bsizeZ;
	int* blockLengths = (int*)malloc(sizeof(int)*bsizeY*bsizeZ);
	MPI_Aint* blockDisplacements = (MPI_Aint*)malloc(sizeof(MPI_Aint)*bsizeY*bsizeZ);
	for (k = 0; k < bsizeZ; ++k) {
		for (j = 0; j < bsizeY; ++j) {
			blockLengths[k*bsizeY + j] = bsizeX * charPerDouble;
			blockDisplacements[k*bsizeY + j] = (j * gsizeX + k * gsizeX*bsizeY*gsizeY)*charPerDouble*bsizeX;
			//printf("%d ", blockDisplacements[k*bsizeY + j]);
		}
		//printf("\n");
	}
	MPI_Type_create_hindexed(blockCount, blockLengths, blockDisplacements, MPI_CHAR, &newHindexedType);
	MPI_Type_commit(&newHindexedType);

	fprintf(stderr, "Writing to file\n");
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
	fprintf(stderr, "End\n");

	free(charBuff);
	free(blockLengths);
	free(blockDisplacements);
	free(Udata);
	free(Unext);
	//if (id == 0) {
		//if (bsizeX*bsizeY*bsizeX*gsizeX*gsizeZ*gsizeY < 400) {
		//	fprintf(stderr, "%d %d %d %d %d %d\n", gsizeX, gsizeY, gsizeZ, bsizeX, bsizeY, bsizeZ);
		//	fprintf(stderr, "%f %f %f %f %f %f %f %f %f %f %f\n", lx, ly, lz, epsilon, Udown, Uup,  Uleft, Uright,  Ufront, Uback, U0);
		//}
	//}

	return 0;
}
