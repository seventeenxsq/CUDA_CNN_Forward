#include <stdio.h>
#include"device_functions.h"
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>
#include <time.h>


/*
函数申明 :
1. __global__ conv_step1( )
这个函数将卷积核和原数组做累乘
再放置到目标数组中去 block为一维，因为表示层级

2. __global__ conv_step2( )
此函数将上述累乘的函数做累加形成完整的卷积操作

3. __global__ pool()  池化操作只要单个过程函数即可

4. __global__ FC()  全连接层的矩阵乘法

*/
#define TILE_SIZE 32
#define Kernel_size 3

__global__ void conv_step1(float *step1_out, float * feature_in,
	float * con_core_in, int featuremap_size, int coresize);

//第二步是做累加，与卷积核无关了
__global__ void conv_step2(float * feature_out,float * step1_out, 
									int core_num, int core_layers, int step1_out_featuremap_size);

__global__ void pool(float *feature_out, float *feature_in, int featuremap_in_size, int poolling_size);

__global__ void FC_SharedMem(float *featuremap_in, float *weight, float *feature_out, int feature_in_size);

__global__ void conv_step1_new(float *step1_out, float * feature_in,
	float * con_core_in, int featuremap_size, int coresize, int layers_in_featuremap);

__global__ void conv_step2_new(float * feature_out, float * step1_out,
		int core_layers, int step1_out_featuremap_size);

__global__ void pool_new(float *feature_out, float *feature_in, int featuremap_in_size, int poolling_size);

