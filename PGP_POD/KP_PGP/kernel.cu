

#include "cuda.h"
#include "PGP_KP_support_functions.h"
#include "time.h"
#include "mpi.h"
//Тетраэдр - правильная треугольная пирамида
//Гексаэдр ака КУБ
//Октаэдр - 3Д ромб aka кристал

int LAUNCH_ARGS = 0;

__constant__ vec3 GPU_Lights[4];
__constant__ vec3 GPU_LightsColours[4];
__constant__ int GPU_LightsAmount;


__constant__ float GPU_AMBIENT = 0.3;
__constant__ float GPU_DIFFUSE = 0.55;
__constant__ float GPU_REFLECTION = 0.15;


__global__ void printmemorystate(uchar4* data,int sz) {
	if (threadIdx.x == 0) {
		printf("Memory State:\n");
		for (int i = 0; i < min(sz,100); ++i) {
			printf("%d: ", i);
			printUchar4(data[i]);
		}
	}
}

__global__ void printconstants() {
	if (threadIdx.x == 0) {
		printf("GPU_LightsAmount: %d\n", GPU_LightsAmount);
		for (int i = 0; i < GPU_LightsAmount; ++i) {
			printf("Light #%d\n", i);
			GPU_Lights[i].printToStdout();
			GPU_LightsColours[i].printToStdout();
		}
	}
}

texture<float4, 2, cudaReadModeElementType> tex;

__global__ void SSAA(float4 *out, int w, int h, int resW, int resH) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	int multiplier = (w / resW) * (h / resH);
	float4 p;
	for (y = idy; y < h; y += offsety)
		for (x = idx; x < w; x += offsetx) {
			p = tex2D(tex, x, y);
			atomicAdd(&out[y / (h / resH) * resW + x / (w / resW)].x, p.x / multiplier);
			atomicAdd(&out[y / (h / resH) * resW + x / (w / resW)].y, p.y / multiplier);
			atomicAdd(&out[y / (h / resH) * resW + x / (w / resW)].z, p.z / multiplier);
		}
}

void CPU_SSAA(float4* in, float4* out, int w, int h, int resW, int resH) {
	int multiplier = (w / resW) * (h / resH);
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			out[y / (h / resH) * resW + x / (w / resW)].x += in[y*w + x].x / multiplier;
			out[y / (h / resH) * resW + x / (w / resW)].y += in[y*w + x].y / multiplier;
			out[y / (h / resH) * resW + x / (w / resW)].z += in[y*w + x].z / multiplier;
		}
	}
}



__device__ vec3 GPUray(vec3 pos, vec3 dir, trig* trigs, int trigsAmount, int recursion) {
	int k, k_min = -1;
	vec3 e1, e2, p, t, q;
	float ts_min;
	for (k = 0; k < trigsAmount; k++) {
		e1 = diff(trigs[k].b, trigs[k].a);  
		e2 = diff(trigs[k].c, trigs[k].a); 
		p = prod(dir, e2);
		float div = dot(p, e1);
		if (fabs(div) < 1e-10)
			continue;
		t = diff(pos, trigs[k].a);
		float u = dot(p, t) / div;
		if (u < 0.0 || u > 1.0)
			continue;
		q = prod(t, e1);
		float v = dot(q, dir) / div;
		if (v < 0.0 || v + u > 1.0)
			continue;
		float ts = dot(q, e2) / div;
		if (ts < 0.0)
			continue;
		if (k_min == -1 || ts < ts_min) {
			k_min = k;
			ts_min = ts;
		}
	}
	if ((recursion == 1)&&(k_min==-1)) { 
		return {-1,-1,-1}; // Источник света виден
	}
	if (k_min == -1)
		return { 0, 0, 0};
	if (recursion > 0) {
		return trigs[k_min].color;
	}

	vec3 z = add(pos, multByNum(dir, ts_min - (float)(0.0001))); // Из-за пограшностей точка пересечения могла уйти под треугольник
	e2 = diff(trigs[k_min].a, trigs[k_min].b);
	e1 = diff(trigs[k_min].b, trigs[k_min].c);
	vec3 resColour = multByNum(trigs[k_min].color, GPU_AMBIENT);
	p = norm(prod(e1, e2)); //Нормаль треугольника
	vec3 tmp;
	if (GPU_DIFFUSE > 1e-5) {
		for (int i = 0; i < GPU_LightsAmount; ++i) {
			t = norm(diff(GPU_Lights[i], z)); //Нормир. верктор к источнику света
			if (vec3Eq(GPUray(z, t, trigs, trigsAmount, 1), { -1,-1,-1 })) {
				float r = modulus(GPU_Lights[i], z); // расстояние до лампы
				//printf("Diffuse light in effect\n");
				tmp = multByNum(GPU_LightsColours[i], GPU_DIFFUSE*min(1, LIGHT_FADING_RADIUS / max(r, 0.000001)));
				resColour = addlimited(resColour, tmp);
			}
		}
	}
	if (GPU_REFLECTION > 1e-5) {
		if (dot(p, dir) > 0) {
			p = multByNum(p, -1);
		}

		tmp = GPUray(z, diff(dir, multByNum(multByNum(p, dot(dir, p)), 2)), trigs, trigsAmount, 2);
		if (abs(trigs[k_min].reflect) > EQ_EPSILON) {

			tmp = multByNum(tmp, GPU_REFLECTION*trigs[k_min].reflect);
			resColour = addlimited(resColour, tmp);

		}
	}
	return resColour;
}

__global__ void GPURender(vec3 Camera, vec3 bx, vec3 by, vec3 bz, int w, int h, float dw, float dh, float z, vec3* data, trig* trigs, int trigsAmount) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	int idy = blockDim.y*blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x*gridDim.x;
	int offsety = blockDim.y*gridDim.y;
	for (int j = idy; j < h; j+=offsety) {
		for (int i = idx; i < w; i+=offsetx) {
			vec3 v = { (float)(-1.0) + dw * i, ((float)(-1.0) + dh * j) * h / w, z };
			//Перемещение вектора луча в базис камеры
			vec3 dir = mult(bx, by,bz, v);
			data[(h - 1 - j) * w + i] = GPUray(Camera, norm(dir), trigs, trigsAmount,0);
		}
	}
}


int main(int argc, char* argv[]) {
	/*vec3 test;
	test.x = 1;
	test.y = 2;
	test.z = 3;
	printf("Size: %d Addresses:%X %X %X %X", sizeof(test), &test, &test.x, &test.y, &test.z);
	return 0;*/
	int frames, frameH, frameW, frameAngle;
	char outputPath[256];

	float r0c, z0c, phi0c, Arc, Azc, omegarc, omegazc, omegaphic, prc, pzc;
	float r0n, z0n, phi0n, Arn, Azn, omegarn, omegazn, omegaphin, prn, pzn;

	vec3 center1, center2, center3;
	vec3 colour1, colour2, colour3;
	float radius1, radius2, radius3;
	float reflect1, reflect2, reflect3;
	float transp1, transp2, transp3;
	float lights1, lights2, lights3;

	vec3 floor[4];
	char floorTexturePath[128];
	vec3 floorColour;
	float floorReflect;


	int maxRecursionDepth;
	int raysPerPixelSqrt;

	int i, j;// , k;

	int numproc, id;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	MPI_Barrier(MPI_COMM_WORLD);
	//printf("ID: %d, TOTAL: %d\n", id, numproc);

	if (id == 0) {

		if (ProcessInputArguments(argc, argv, &LAUNCH_ARGS)) {
			fprintf(stderr, "Encountered error while processing launch aruments. Exiting...\n");
			exit(0);
		};
		if (LAUNCH_ARGS&DEFAULT) {
			PrintKrasivoe();
			return 0;
		}

		//printf("%d\n", LAUNCH_ARGS);
		//FILE* inputFile = stdin;
		FILE* inputFile = fopen("KP_input.txt", "r");
		//Data input
		fscanf(inputFile, "%d", &frames);
		fscanf(inputFile, "%s", outputPath);
		fscanf(inputFile, "%d %d %d", &frameW, &frameH, &frameAngle);
		fscanf(inputFile, "%f %f %f %f %f %f %f %f %f %f", &r0c, &z0c, &phi0c, &Arc, &Azc, &omegarc, &omegazc, &omegaphic, &prc, &pzc);
		fscanf(inputFile, "%f %f %f %f %f %f %f %f %f %f", &r0n, &z0n, &phi0n, &Arn, &Azn, &omegarn, &omegazn, &omegaphin, &prn, &pzn);

		center1.readFromStream(inputFile);
		colour1.readFromStream(inputFile);
		fscanf(inputFile, "%f %f %f %f", &radius1, &reflect1, &transp1, &lights1);


		center2.readFromStream(inputFile);
		colour2.readFromStream(inputFile);
		fscanf(inputFile, "%f %f %f %f", &radius2, &reflect2, &transp2, &lights2);

		center3.readFromStream(inputFile);
		colour3.readFromStream(inputFile);
		fscanf(inputFile, "%f %f %f %f", &radius3, &reflect3, &transp3, &lights3);

		for (i = 0; i < 4; ++i) {
			floor[i].readFromStream(inputFile);
		}
		fscanf(inputFile, "%s", floorTexturePath);
		floorColour.readFromStream(inputFile);
		fscanf(inputFile, "%f", &floorReflect);

		lightsAmount = (int*)malloc(sizeof(int));
		fscanf(inputFile, "%d", lightsAmount);

		Lights = (vec3*)malloc(sizeof(vec3)*(*lightsAmount));
		LightsColours = (vec3*)malloc(sizeof(vec3)*(*lightsAmount));

		for (i = 0; i < *lightsAmount; ++i) {
			Lights[i].readFromStream(inputFile);
			LightsColours[i].readFromStream(inputFile);
		}
		fscanf(inputFile, "%d %d", &maxRecursionDepth, &raysPerPixelSqrt);
	}

	MPI_Bcast(&LAUNCH_ARGS, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Bcast(&frames, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&frameW, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&frameH, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&frameAngle, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&raysPerPixelSqrt, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&maxRecursionDepth, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Bcast(outputPath, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
	MPI_Bcast(floorTexturePath, 128, MPI_CHAR, 0, MPI_COMM_WORLD);

	MPI_Bcast(&r0c, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&z0c, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&phi0c, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Arc, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Azc, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&omegarc, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&omegazc, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&omegaphic, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&prc, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&pzc, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	

	MPI_Bcast(&r0n, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&z0n, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&phi0n, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Arn, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Azn, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&omegarn, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&omegazn, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&omegaphin, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&prn, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&pzn, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);


	MPI_Bcast(&radius1, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&radius2, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&radius3, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&reflect1, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&reflect2, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&reflect3, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&transp1, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&transp2, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&transp3, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lights1, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lights2, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lights3, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	MPI_Bcast(&floorReflect, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);


	MPI_Bcast(&center1, 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&center2, 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&center3, 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&colour1, 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&colour2, 3, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&colour3, 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

	//for (i = 0; i < 4; ++i) {
	MPI_Bcast(floor, 12, MPI_FLOAT, 0, MPI_COMM_WORLD);
	//}

	MPI_Bcast(&floorColour, 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

	if (id != 0) {
		lightsAmount = (int*)malloc(sizeof(int));
	}

	MPI_Bcast(lightsAmount, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (id != 0) {
		Lights = (vec3*)malloc(sizeof(vec3)*(*lightsAmount));
		LightsColours = (vec3*)malloc(sizeof(vec3)*(*lightsAmount));
	}
	
	MPI_Bcast(Lights, 3* (*lightsAmount), MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(LightsColours, 3 * (*lightsAmount), MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);


	int framesPerProccess = frames / numproc;
	int framesLeft = frames % numproc;
	int startingFrame = framesPerProccess * id;
	startingFrame += (id >= framesLeft ? framesLeft : id);
	int endingFrame = startingFrame + framesPerProccess + (id >= framesLeft ? 0 : 1);
	//printf("FPP: %d, FL:%d Start:%d Last:%d\n", framesPerProccess, framesLeft,startingFrame,endingFrame);
	MPI_Barrier(MPI_COMM_WORLD);

	//---------------------------------------------------------------------------------------------------

	clock_t start, end;
	
	int renderH = frameH * raysPerPixelSqrt;
	int renderW = frameW * raysPerPixelSqrt;
	int trigsAmount = 2 + 4 + 12 + 8;
	trig* trigs = BuildSpace(floor,floorColour,floorReflect,center1,colour1,radius1,reflect1, center2, colour2, radius2, reflect2, center3, colour3, radius3,reflect3);
	//Дальше цикл по кадрам уже
	float t;
	vec3 cameraPoint, cameraDirection;
	char outputBuff[264];
	if (frames > 1e6) {
		fprintf(stderr, "ERROR: string buffer for output file name is too small :(\nReduce number of frames please\n");
		free(Lights);
		free(LightsColours);
		return 0;
	}


	vec3* pixels;// = (vec3*)malloc(sizeof(vec3)*renderW*renderH);
	float4* f4pixels;// = (float4*)malloc(sizeof(float4)*renderW*renderH);
	uchar4* ssaaPixels;// = (uchar4*)malloc(sizeof(uchar4)*frameW*frameH);
	float4* floatPixels;// = (float4*)malloc(sizeof(float4)*frameW*frameH);

	float lightDz = (float)1/frames;
	
	LightsColours[1].y -= lightDz * startingFrame;
	LightsColours[1].z -= lightDz * startingFrame;

	cudaEvent_t Gstart, Gend;
	float gputime;

	vec3* GPUpixels;
	float4* GPUfloatpixels;
	trig* GPUtrigs;
	cudaArray *arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<float4>();

	int deviceCount;
	CSC(cudaGetDeviceCount(&deviceCount));
	if (deviceCount != 1) {
		CSC(cudaSetDevice(id & deviceCount));
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (LAUNCH_ARGS&GPU) {

		CSC(cudaMalloc(&GPUpixels, sizeof(vec3)*renderW*renderH));
		CSC(cudaMalloc(&GPUtrigs, sizeof(trig)*trigsAmount));
		CSC(cudaMemcpy(GPUtrigs, trigs, sizeof(trig)*trigsAmount, cudaMemcpyHostToDevice));
		//CSC(cudaMemcpy(GPUpixels, pixels, sizeof(vec3)*renderW*renderH, cudaMemcpyHostToDevice));
		
		CSC(cudaEventCreate(&Gstart));
		CSC(cudaEventCreate(&Gend));
		
		CSC(cudaMallocArray(&arr, &ch, renderW, renderH));
		
		tex.addressMode[0] = cudaAddressModeClamp;
		tex.addressMode[1] = cudaAddressModeClamp;
		tex.channelDesc = ch;
		tex.filterMode = cudaFilterModePoint;
		tex.normalized = false;
		
		CSC(cudaBindTextureToArray(tex, arr, ch));
		
		CSC(cudaMalloc(&GPUfloatpixels, sizeof(float4)*frameW*frameH));
	

		CSC(cudaMemcpyToSymbol(GPU_Lights, Lights, sizeof(vec3) * (*lightsAmount), 0, cudaMemcpyHostToDevice));
		CSC(cudaMemcpyToSymbol(GPU_LightsColours, LightsColours, sizeof(vec3) * (*lightsAmount), 0, cudaMemcpyHostToDevice));
		CSC(cudaMemcpyToSymbol(GPU_LightsAmount, lightsAmount, sizeof(int), 0, cudaMemcpyHostToDevice));

	}

	MPI_Barrier(MPI_COMM_WORLD);
	vec3 bx, by, bz;
	float dw, dh, z;
	clock_t globStart, globEnd;
	globStart = clock();
	if (LAUNCH_ARGS&CPU) {
		#pragma omp parallel for private(i,t,cameraPoint,cameraDirection,pixels,f4pixels,ssaaPixels,floatPixels) schedule(dynamic)
		for (int i = startingFrame; i < endingFrame; ++i) {
			//printf("ID:%d,FRAME:%d\n", id, i);
			pixels = (vec3*)malloc(sizeof(vec3)*renderW*renderH);
			f4pixels = (float4*)malloc(sizeof(float4)*renderW*renderH);
			ssaaPixels = (uchar4*)malloc(sizeof(uchar4)*frameW*frameH);
			floatPixels = (float4*)malloc(sizeof(float4)*frameW*frameH);
			t = i * (float)(2 * M_PI / frames);

			cameraPoint = { (r0c + Arc * sin((omegarc*t + prc)))*cos((phi0c + omegaphic * t)),
							(r0c + Arc * sin((omegarc*t + prc)))*sin((phi0c + omegaphic * t)),
							z0c + Azc * sin((omegazc*t + pzc))
			};
			cameraDirection = { (r0n + Arn * sin((omegarn*t + prn)))*cos((phi0n + omegaphin * t)),
								(r0n + Arn * sin((omegarn*t + prn)))*sin((phi0n + omegaphin * t)),
								z0n + Azn * sin((omegazn*t + pzn))
			};
			//printf("Frame: %d\n", i);


			//printf("Running CPU render %d\n", i);
			start = clock();
			render(cameraPoint, cameraDirection, renderW, renderH, (float)frameAngle, pixels, trigs, 2 + 4 + 12 + 8);
			end = clock();
			printf("ID: %d Frame: %d CPU: %ld ms\n", id, i, end - start);

			convertVec3ToFloat4(pixels, f4pixels, renderW*renderH);

			for (j = 0; j < frameW*frameH; ++j) {
				floatPixels[j] = { 0,0,0,0 };
			}

			//printf("Starting CPU SSAA %d\n", i);
			start = clock();
			CPU_SSAA(f4pixels, floatPixels, renderW, renderH, frameW, frameH);
			end = clock();
			printf("ID: %d Frame: %d CPU SSAA: %ld ms\n", id, i, end - start);


			for (j = 0; j < frameW*frameH; ++j) {
				ssaaPixels[j] = UcharFromNormalFloat4(floatPixels[j]);
			}

			sprintf(outputBuff, outputPath, i);
			printf("ID: %d Frame: %d - %s\n", id, i, outputBuff);

			FILE *out = fopen(outputBuff, "wb");
			if (out == NULL) {
				fprintf(stderr, "ERROR: Can't create output file :(\n");
				break;
			}
			fwrite(&frameW, sizeof(int), 1, out);
			fwrite(&frameH, sizeof(int), 1, out);
			fwrite(ssaaPixels, sizeof(uchar4), frameW * frameH, out);
			fclose(out);

			free(pixels);
			free(f4pixels);
			free(ssaaPixels);
			free(floatPixels);
		}
	}
	if (LAUNCH_ARGS&GPU) {
		pixels = (vec3*)malloc(sizeof(vec3)*renderW*renderH);
		f4pixels = (float4*)malloc(sizeof(float4)*renderW*renderH);
		ssaaPixels = (uchar4*)malloc(sizeof(uchar4)*frameW*frameH);
		floatPixels = (float4*)malloc(sizeof(float4)*frameW*frameH);
		for (int i = startingFrame; i < endingFrame; ++i) {
			//printf("ID:%d,FRAME:%d\n", id, i);
			t = i * (float)(2 * M_PI / frames);

			cameraPoint = { (r0c + Arc * sin((omegarc*t + prc)))*cos((phi0c + omegaphic * t)),
							(r0c + Arc * sin((omegarc*t + prc)))*sin((phi0c + omegaphic * t)),
							z0c + Azc * sin((omegazc*t + pzc))
			};
			cameraDirection = { (r0n + Arn * sin((omegarn*t + prn)))*cos((phi0n + omegaphin * t)),
								(r0n + Arn * sin((omegarn*t + prn)))*sin((phi0n + omegaphin * t)),
								z0n + Azn * sin((omegazn*t + pzn))
			};
			//printf("Frame: %d\n", i);

			dw = (float)(2.0 / (renderW - 1.0));
			dh = (float)(2.0 / (renderH - 1.0));
			z = (float)(1.0 / tan(frameAngle * M_PI / 360.0));
			//Базис в точке камеры
			bz = norm(diff(cameraDirection, cameraPoint)); // Направление камеры
			bx = norm(prod(bz, { 0.0, 0.0, 1.0 })); // В небо
			by = norm(prod(bx, bz)); // Остаток

			CSC(cudaEventRecord(Gstart));

			GPURender << <dim3(256, 256), dim3(16, 16) >> > (cameraPoint, bx, by, bz, renderW, renderH, dw, dh, z, GPUpixels, GPUtrigs, trigsAmount);

			CSC(cudaGetLastError());
			//printmemorystate << <1, 1 >> > (GPUpixels, renderH*renderW);
			CSC(cudaGetLastError());
			CSC(cudaEventRecord(Gend));
			CSC(cudaEventSynchronize(Gend));
			CSC(cudaEventElapsedTime(&gputime, Gstart, Gend));
			printf("ID: %d Frame: %d GPU: %f ms\n",id,i, gputime); //end - start);
			CSC(cudaMemcpy(pixels, GPUpixels, sizeof(vec3)*renderW*renderH, cudaMemcpyDeviceToHost));

			convertVec3ToFloat4(pixels, f4pixels, renderW*renderH);

			for (j = 0; j < frameW*frameH; ++j) {
				floatPixels[j] = { 0,0,0,0 };
			}

			CSC(cudaEventRecord(Gstart));

			CSC(cudaMemcpyToArray(arr, 0, 0, f4pixels, sizeof(float4) * renderW * renderH, cudaMemcpyHostToDevice));
			CSC(cudaMemcpy(GPUfloatpixels, floatPixels, sizeof(float4)*frameW*frameH, cudaMemcpyHostToDevice));

			SSAA << <dim3(32, 32), dim3(16, 16) >> > (GPUfloatpixels, renderW, renderH, frameW, frameH);

			CSC(cudaMemcpy(floatPixels, GPUfloatpixels, sizeof(float4)*frameW*frameH, cudaMemcpyDeviceToHost));

			CSC(cudaGetLastError());
			CSC(cudaEventRecord(Gend));
			CSC(cudaEventSynchronize(Gend));
			CSC(cudaEventElapsedTime(&gputime, Gstart, Gend)); 
			printf("ID: %d Frame: %d SSAA+memcpy: %f ms\n", id, i, gputime);


			for (j = 0; j < frameW*frameH; ++j) {
				ssaaPixels[j] = UcharFromNormalFloat4(floatPixels[j]);
			}
			sprintf(outputBuff, outputPath, i);
			printf("ID: %d Frame: %d - %s\n", id, i, outputBuff);

			FILE *out = fopen(outputBuff, "wb");
			if (out == NULL) {
				fprintf(stderr, "ERROR: Can't create output file :(\n");
				break;
			}
			fwrite(&frameW, sizeof(int), 1, out);
			fwrite(&frameH, sizeof(int), 1, out);
			fwrite(ssaaPixels, sizeof(uchar4), frameW * frameH, out);
			fclose(out);

			LightsColours[1].y -= lightDz;
			LightsColours[1].z -= lightDz;
			//LightsColours[1].printToStdout();
			CSC(cudaMemcpyToSymbol(GPU_LightsColours, LightsColours, sizeof(vec3) * (*lightsAmount), 0, cudaMemcpyHostToDevice));

		}
		free(pixels);
		free(f4pixels);
		free(ssaaPixels);
		free(floatPixels);
	}

	globEnd = clock();
	printf("ID: %d. Summary: %ld ms",id, globEnd - globStart);
	free(trigs);
	free(Lights);
	free(LightsColours);
	if (LAUNCH_ARGS&GPU) {
		CSC(cudaUnbindTexture(tex));
		CSC(cudaFreeArray(arr));
		CSC(cudaFree(GPUfloatpixels));
		CSC(cudaFree(GPUpixels));
		CSC(cudaFree(GPUtrigs));
		//CSC(cudaFree(GPUfloatpixels));
	}
	MPI_Finalize();
	//CSC(cudaFree(GPUtrigs));
	//CSC(cudaFree(GPUpixels));
	return 0;
}