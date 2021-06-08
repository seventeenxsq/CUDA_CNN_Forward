#include <stdio.h>
#include"device_functions.h"
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>
#include <time.h>


/*
�������� :
1. __global__ conv_step1( )
�������������˺�ԭ�������۳�
�ٷ��õ�Ŀ��������ȥ blockΪһά����Ϊ��ʾ�㼶

2. __global__ conv_step2( )
�˺����������۳˵ĺ������ۼ��γ������ľ������

3. __global__ pool()  �ػ�����ֻҪ�������̺�������

4. __global__ FC()  ȫ���Ӳ�ľ���˷�

*/
#define TILE_SIZE 32
#define Kernel_size 3

__global__ void conv_step1(float *step1_out, float * feature_in,
	float * con_core_in, int featuremap_size, int coresize);

//�ڶ��������ۼӣ��������޹���
__global__ void conv_step2(float * feature_out,float * step1_out, 
									int core_num, int core_layers, int step1_out_featuremap_size);

__global__ void pool(float *feature_out, float *feature_in, int featuremap_in_size, int poolling_size);

__global__ void FC_SharedMem(float *featuremap_in, float *weight, float *feature_out, int feature_in_size);

__global__ void conv_step1_new(float *step1_out, float * feature_in,
	float * con_core_in, int featuremap_size, int coresize, int layers_in_featuremap);

__global__ void conv_step2_new(float * feature_out, float * step1_out,
		int core_layers, int step1_out_featuremap_size);

__global__ void pool_new(float *feature_out, float *feature_in, int featuremap_in_size, int poolling_size);

