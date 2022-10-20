#pragma once

#define _USE_MATH_DEFINES
//#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <math.h>
#include "PGP_KP_structs.h"

#define GPU 1
#define CPU 2
#define DEFAULT 4 


#define LIGHT_FADING_RADIUS 4
#define M_PI 3.14159265358979323846
#define ToRads (float)(M_PI/180)
#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

vec3* Lights;
vec3* LightsColours;
int* lightsAmount;

const float CPU_AMBIENT = (float)0.3;
const float CPU_DIFFUSE = (float)0.55;
const float CPU_REFLECTION = (float)0.15;

int ProcessInputArguments(int argc, char* argv[],int* LAUNCH_KEYS) {
	if (argc == 1) {
		*LAUNCH_KEYS |= GPU;
		return 0;
	}
	if (argc > 2) {
		fprintf(stderr, "ERROR: Too many launch keys. Maximum - 1.\n");
		return 1;
	}
	if ((strcmp("-gpu", argv[1]) == 0) || (strcmp("-GPU", argv[1]) == 0)) {
		*LAUNCH_KEYS |= GPU;
		//printf("Using: GPU\n");
		return 0;
	}
	if ((strcmp("-cpu", argv[1]) == 0) || (strcmp("-CPU", argv[1]) == 0)) {
		*LAUNCH_KEYS |= CPU;
		//printf("Using: CPU(OpenMP)\n");
		return 0;
	}
	if ((strcmp("-default", argv[1]) == 0) || (strcmp("-Default", argv[1]) == 0) || (strcmp("-DEFAULT", argv[1]) == 0)) {
		*LAUNCH_KEYS |= DEFAULT;
		return 0;
	}
	fprintf(stderr, "ERROR: Unknown launch key. Use one of -gpu -cpu -default\n");
	return 1;
}

vec3 RotatePointXY(vec3 point, float angle) {
	return {point.x*cosf(angle*ToRads) - point.y * sinf(angle*ToRads),
			point.x*sinf(angle*ToRads) + point.y * cosf(angle*ToRads),
			point.z };
}

void CreateTetrahedron(vec3 center,  float rad, vec3* result) {
	//vec3 result[4];
	vec3 tmp;
	result[0] = { center.x, center.y, center.z + rad };
	result[1] = { center.x, center.y + 2 * sqrtf(2)*rad / 3, center.z - rad / 3 };
	tmp = RotatePointXY( { 0,2 * sqrtf(2)*rad / 3,center.z - rad / 3 }, 120);
	result[2] = { center.x + tmp.x, center.y + tmp.y, tmp.z };
	result[3] = { center.x - tmp.x, center.y + tmp.y, tmp.z };
	//return result;
}

void CreateOctahedron(vec3 center, float rad, vec3* result) {
	//vec3 result[4];
	result[0] = { center.x, center.y, center.z + rad };
	result[1] = { center.x, center.y, center.z - rad };
	result[2] = { center.x-rad, center.y, center.z};
	result[3] = { center.x+rad, center.y, center.z};
	result[4] = { center.x, center.y-rad, center.z};
	result[5] = { center.x, center.y+rad, center.z};
	//return result;
}
#define CreateCrystal CreateOctahedron

void CreateCube(vec3 center, float rad, vec3* result) {
	//vec3 result[4];
	float offset = rad / sqrtf(3);
	result[0] = { center.x - offset, center.y - offset, center.z - offset};
	result[1] = { center.x - offset, center.y + offset, center.z - offset };
	result[2] = { center.x + offset, center.y + offset, center.z - offset };
	result[3] = { center.x + offset, center.y - offset, center.z - offset };
	result[4] = { center.x - offset, center.y - offset, center.z + offset };
	result[5] = { center.x - offset, center.y + offset, center.z + offset };
	result[6] = { center.x + offset, center.y + offset, center.z + offset };
	result[7] = { center.x + offset, center.y - offset, center.z + offset };
	//return result;
}

trig* BuildSpace(vec3 floor[4], vec3 floorColour, float floorReflect,
				vec3 tetra, vec3 tetraColour, float tetraRad, float tetraReflect,
				vec3 cube, vec3 cubeColour, float cubeRad, float cubeReflect,
				vec3 crystal, vec3 crystalColour, float crystalRad, float crystalReflect ) {
	trig* trigs = (trig*)malloc(sizeof(trig)*(2 + 4 + 12 + 8));//2 - пол, 4 - тетраэдр, 12 - куб, 8 - Октаэдр(кристал)
	trigs[0] = { floor[0],floor[1],floor[2],floorColour, floorReflect };
	trigs[1] = { floor[0],floor[3],floor[2],floorColour, floorReflect };
	//Tetraedr
	vec3* tet = (vec3*)malloc(sizeof(vec3) * 4);
	CreateTetrahedron(tetra, tetraRad, tet);
	trigs[2] = {tet[0],tet[1],tet[2], tetraColour, tetraReflect };
	trigs[3] = { tet[0],tet[1],tet[3], tetraColour, tetraReflect };
	trigs[4] = { tet[0],tet[2],tet[3], tetraColour, tetraReflect };
	trigs[5] = { tet[1],tet[2],tet[3], tetraColour, tetraReflect };
	free(tet);
	
	tet = (vec3*)malloc(sizeof(vec3) * 6);
	CreateCrystal(crystal, crystalRad, tet);
	trigs[6] = { tet[0],tet[2],tet[4],(crystalColour), crystalReflect };
	trigs[7] = { tet[0],tet[2],tet[5],(crystalColour), crystalReflect };
	trigs[8] = { tet[0],tet[3],tet[4],(crystalColour), crystalReflect };
	trigs[9] = { tet[0],tet[3],tet[5],(crystalColour), crystalReflect };

	trigs[10] = { tet[1],tet[2],tet[4],(crystalColour), crystalReflect };
	trigs[11] = { tet[1],tet[2],tet[5],(crystalColour), crystalReflect };
	trigs[12] = { tet[1],tet[3],tet[4],(crystalColour), crystalReflect };
	trigs[13] = { tet[1],tet[3],tet[5],(crystalColour), crystalReflect };
	free(tet);

	tet = (vec3*)malloc(sizeof(vec3) * 8);
	CreateCube(cube, cubeRad, tet);
	//Боковые грани, по 2 треугольника
	for (int i = 0; i < 4; ++i) {
		trigs[14 + 2 * i] = {tet[i],tet[4+i],tet[4+(i+1)%4],cubeColour, cubeReflect };
		trigs[14 + 2 * i + 1] = { tet[i],tet[(1 + i)%4],tet[4 + (i + 1) % 4],cubeColour,cubeReflect };
	}
	//Нижняя и верхняя грани
	for (int i = 0; i < 8; i += 4) {
		trigs[22 + ((int)(i / 4) * 2)] = { tet[i],tet[i + 1],tet[i + 2],cubeColour,cubeReflect };
		trigs[22 + ((int)(i / 4) * 2)+1] = { tet[i],tet[i + 3],tet[i + 2],cubeColour,cubeReflect };
	}
	free(tet);
	return trigs;
}

vec3 ray(vec3 pos, vec3 dir, trig* trigs, int trigsAmount, int recursion) {
	int k, k_min = -1;
	float ts_min;
	vec3 e1, e2, p, t, q;
	for (k = 0; k < trigsAmount; k++) {
		e1 = diff(trigs[k].b, trigs[k].a); //u 
		e2 = diff(trigs[k].c, trigs[k].a); //v
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
	if ((recursion == 1) && (k_min == -1)) {
		return { -1,-1,-1 };
	}
	if (k_min == -1)
		return { 0, 0, 0 };
	if (recursion > 0) {
		return trigs[k_min].color;
	}
	vec3 z = add(pos, multByNum(dir, ts_min - (float)(0.0001))); // Из-за пограшностей точка пересечения могла уйти под треугольник
	e2 = diff(trigs[k_min].a, trigs[k_min].b);
	e1 = diff(trigs[k_min].b, trigs[k_min].c);

	vec3 resColour = multByNum(trigs[k_min].color, CPU_AMBIENT);
	p = norm(prod(e1, e2)); //Нормаль треугольника

	vec3 tmp;
	if (CPU_DIFFUSE > 1e-5) {
		for (int i = 0; i < *lightsAmount; ++i) {
			t = norm(diff(Lights[i], z)); //Нормир. верктор к источнику света
			if (vec3Eq(ray(z, t, trigs, trigsAmount, 1), { -1,-1,-1 })) {
				float r = modulus(Lights[i], z); // расстояние до лампы

				tmp = multByNum(LightsColours[i], CPU_DIFFUSE*min(1, LIGHT_FADING_RADIUS / max(r, (float)(0.000001))));
				resColour = addlimited(resColour, tmp);
			}
		}
	}
	if (CPU_REFLECTION > 1e-5) {
		if (dot(p, dir) > 0) {
			p = multByNum(p, -1);
		}

		tmp = ray(z, diff(dir, multByNum(multByNum(p, dot(dir, p)), 2)), trigs, trigsAmount, 2);
		if (abs(trigs[k_min].reflect) > EQ_EPSILON) {

			tmp = multByNum(tmp, CPU_REFLECTION*trigs[k_min].reflect);
			resColour = addlimited(resColour, tmp);
		}
	}
	return resColour;
}

void render(vec3 pc, vec3 pv, int w, int h, float angle, vec3 *data, trig* trigs, int trigsAmount) {
	int i, j;
	float dw = (float)(2.0 / (w - 1.0));
	float dh = (float)(2.0 / (h - 1.0));
	float z = (float)(1.0) / tanf(angle * (float)( M_PI /360.0));
	//Базис в точке камеры
	vec3 bz = norm(diff(pv, pc)); // Направление камеры
	vec3 bx = norm(prod(bz, { 0.0, 0.0, 1.0 })); // В небо
	vec3 by = norm(prod(bx, bz)); // Остаток
	vec3 v, dir;
	for (i = 0; i < w; i++)
		for (j = 0; j < h; j++) {
			v = { (float)(-1.0) + dw * i, ((float)(-1.0) + dh * j) * h / w, z };
			//Перемещение вектора луча в базис камеры
			dir = mult(bx, by, bz, v);
			data[(h - 1 - j) * w + i] = ray(pc, norm(dir),trigs,trigsAmount,0);
		}
}

void convertVec3ToFloat4(vec3* in, float4* out,int sz) {
	vec3 tmp;
	for (int i = 0; i < sz; ++i) {
		tmp = in[i];
		out[i] = make_float4(tmp.x, tmp.y, tmp.z, 1.0);
	}
}

uchar4 UcharFromNormalFloat4(float4 vec) {
	uchar4 res;
	res.x = (uchar)(min(vec.x,1) * 255);
	res.y = (uchar)(min(vec.y,1) * 255);
	res.z = (uchar)(min(vec.z,1) * 255);
	res.w = 255;
	return res;
}

void PrintKrasivoe() {
	printf("128\n");
	printf("out/img_%%d.data\n");
	printf("728 728 120\n");
	
	printf("5.0 5.0 0.0 2.0 1.0 2.0 6.0 1.0 0.0 0.0\n");
	printf("2.0 0.0 0.0 0.5 0.1 1.0 4.0 1.0 0.0 0.0\n");
	
	printf("3.0 0.0 0.0 1.0 0.0 0.0 1.0 0.5 0.1 10\n");
	printf("0.0 3.0 0.0 0.0 1.0 0.0 1.0 0.6 0.2 5\n");
	printf("0.0 -2.0 2.5 0.0 0.0 1.0 1.3 0.4 0.3 2\n");

	printf("-16.0 -16.0 -4.5\n");
	printf("-16.0 16.0 -4.5\n");
	printf("16.0 16.0 -4.5\n");
	printf("16.0 -16.0 -4.5\n");
	printf("~/floor.data\n");
	printf("1.0 1.0 1.0\n");
	printf("0.8\n");


	printf("2\n");
	printf("-7.0 0.0 7.0 0.9 0.9 0.9\n");
	printf("0.0 0.0 3.0 0.6 0.6 0.9\n");
	printf("10 2\n");
}