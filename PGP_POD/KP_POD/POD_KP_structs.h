#pragma once
#include <stdio.h>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))


typedef unsigned char uchar;
#define EQ_EPSILON 1e-10

struct vec3 {
	float x;
	float y;
	float z;

	void readFromStream(FILE* strm) {
		fscanf(strm,"%f", &x);
		fscanf(strm,"%f", &y);
		fscanf(strm,"%f", &z);
	}
	void printToStream(FILE* strm) {
		fprintf(strm,"%f %f %f\n", x, y, z);
	}

	__host__ __device__ void printToStdout() {
		printf("%f %f %f\n", x, y, z);
	}
};



__host__ __device__ float dot(vec3 a, vec3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float modulus(vec3 a, vec3 b) {
	return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) + (a.z - b.z)*(a.z - b.z));
}
__host__ __device__ vec3 prod(vec3 a, vec3 b) {
	return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

__host__ __device__ vec3 norm(vec3 v) {
	float l = sqrt(dot(v, v));
	return { v.x / l, v.y / l, v.z / l };
}

__host__ __device__ vec3 diff(vec3 a, vec3 b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

__host__ __device__ vec3 add(vec3 a, vec3 b) {
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}


__host__ __device__ vec3 addlimited(vec3 a, vec3 b) {
	return { min(a.x + b.x,1), min(a.y + b.y, 1), min(a.z + b.z,1) };
}

__host__ __device__ vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v) {
	return { a.x * v.x + b.x * v.y + c.x * v.z,
				a.y * v.x + b.y * v.y + c.y * v.z,
				a.z * v.x + b.z * v.y + c.z * v.z };
}
__host__ __device__ vec3 multByNum(vec3 a, float b) {
	return { a.x*b, a.y*b,a.z*b };
}
__host__ __device__ bool vec3Eq(vec3 a, vec3 b) {
	return ((fabs(a.x - b.x) < EQ_EPSILON) && (fabs(a.z - b.z) < EQ_EPSILON) && (fabs(a.z - b.z) < EQ_EPSILON));
}

__host__ __device__ bool uchar4Eq(uchar4 a, uchar4 b) {
	return (a.x==b.x && a.y==b.y && a.z==b.z && a.w == b.w);
}

//not in use
__host__ __device__ uchar4 multUchar4ByNum(uchar4 a, float b) {
	/*if ((b < 0) || (b > 1)) {
		printf( "ERROR: multiplying uchar4 by float out of [0,1]\n");
		return { 0,0,0,0 };
	}*/
	return make_uchar4(a.x*b, a.y*b, a.z*b, a.w);// { static_cast<uchar>(a.x*b), static_cast<uchar>(a.y*b), static_cast<uchar>(a.z*b), a.w };
}

__host__ __device__ void printUchar4(uchar4 a){// ,// FILE* fl) {
	printf("%d %d %d %d\n", a.x, a.y, a.z, a.w);
}

//not in use
__host__ __device__ uchar4 addUchar4(uchar4 a, uchar4 b) {
	return { (uchar)min(a.x + b.x,255), (uchar)min(a.y + b.y,255), (uchar)min(a.z + b.z,255), a.w };
}


struct trig {
	vec3 a;
	vec3 b;
	vec3 c;
	vec3 color;
	float reflect;
};

