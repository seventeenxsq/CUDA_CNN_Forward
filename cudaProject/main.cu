#include "cuda_cnn.cuh"

#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\types_c.h>
using namespace cv;
using namespace std;

//我们先来做一个二维的卷积运算
/*
  待卷积矩阵是一个 16*16的矩阵
  卷积核大小 为3
  输出的也是 16*16的矩阵
  threadsperblock(16,16)

  我应该传进去的参数有
  矩阵的地址
  结果矩阵的地址
  卷积核的地址
  卷积核的宽度
  矩阵的场和宽
 */

__global__ void conv(float *in, float *out, float *mask, int maskwidth, int w, int h) {
	// 先要确定索引， 知道我这个线程是在干什么
	int raw = threadIdx.x;
	int col = threadIdx.y;

	int startx = raw - maskwidth / 2;
	int starty = col - maskwidth / 2;

	float tempvalue;

	//卷积内部计算
	for (int i = startx; i < startx + maskwidth; i++)
	{
		for (int j = starty; j < starty + maskwidth; j++)
		{
			//计算tempvalue时需要判定数组越界
			if (i >= 0 && j >= 0 && i < maskwidth && j < maskwidth)
				tempvalue += in[i*w + j] * mask[(i - startx)*maskwidth + (j - starty)];
		}
	}
	out[raw*w + col] = tempvalue;
}

__global__ void multiple2D(float * arr_A, float * arr_B, float * out) {
	
	int raw = threadIdx.x;
	int col = threadIdx.y;

	out[raw * 2 + col] = arr_A[raw * 2 + col] * arr_B[raw * 2 + col];
}

void test_conv_step1(){
	//确定我要开采的feature大小
	int arr_size_3D =3* 5*5;
	//kernel的大小
	int kernel_3d = 10 * 3 * 3 * 3; 
	//step1_out大小
	int step1_out_size = 10 * 3 * 5 * 5;

	/*
	定义数据指针时 我们需要定义两种
	一个是CPU中的数据指针  一种是传到GPU中的数据指针
	为方便我们我们直接都定义一维数组
	*/

	// 首先是cpu中的 定义数组并赋值
	float *feature_in_cpu = (float*)malloc(arr_size_3D * sizeof(float));
	float *kernel_in_cpu = (float*)malloc(kernel_3d * sizeof(float));
	float *feature_out_cpu = (float*)malloc(step1_out_size * sizeof(float));

	////////////////////////////////////////////////////////////////////
	//-------------------  数据赋值部分 -----------------------
	//feature赋值
	for(int layer=0;layer<3;layer++){
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				feature_in_cpu[layer*5*5+i*5+j] = layer+1;
			}
		}
	}

	//kernel 赋值  4维循环
	for(int kernel_num=0;kernel_num<10;kernel_num++){
		for(int layer=0;layer<3;layer++){
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					kernel_in_cpu[(kernel_num*3+layer)*3*3+i*3+j] = kernel_num+1;
				}
			}
		}
	}	

	//-----------------------------数据赋值完成------------------------------
	//////////////////////////////////////////////////////////////////////////

	////输出featrue结果 调试用
	printf(" \n featrue初始化 \n ");
	for(int layer=0;layer<3;layer++){
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				printf("%f\t", feature_in_cpu[layer*5*5+i*5+j]);
			}
		}
		printf("\n");
	}

	printf("--------------------------------------------------------");

	////输出kernel结果 调试用  4维
	printf(" \n kernel的初始化 \n ");
	for(int kernel_num=0;kernel_num<10;kernel_num++){
		for(int layer=0;layer<3;layer++){
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					printf("%f\t", kernel_in_cpu[(kernel_num*3+layer)*3*3+i*3+j]);
				}
			}
			printf("\n\n");
		}
		printf("------------- 第 %d 个卷积核数据 --------------\n", kernel_num);
	}

	printf("\n 初始化一切正常！！！\n");

	//定义GPU的数组指针 define GPU value pointer 
	float *feature_in_gpu, *kernel_in_gpu, *out_gpu;
	//他们都是要传到GPUcache中的数据

	// 在 GPU中 用cudaMalloc开辟内存空间
	//一定要记得传进去 二级指针
	cudaMalloc((void**)&feature_in_gpu, sizeof(float)*arr_size_3D);
	cudaMalloc((void**)&kernel_in_gpu, sizeof(float)*kernel_3d);
	cudaMalloc((void**)&out_gpu, sizeof(float)*step1_out_size);

	// 将CPU中的数据传到 GPU内存中
	cudaMemcpy(feature_in_gpu, feature_in_cpu, sizeof(float)*arr_size_3D, cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_in_gpu, kernel_in_cpu, sizeof(float)*kernel_3d, cudaMemcpyHostToDevice);

	printf("\n 内存COPY完成 \n");
	//输出结果 调试用

	//设计传入的线程  启用线程
	dim3 dimBlock(5, 5);// 线程的形状
	dim3 dimGrid(10, 3);

	//调用CUDA API时我们应该传入的是device的内存指针！！！
	//否则直接就调用不了kernel
	conv_step1<<<dimGrid,dimBlock>>>(out_gpu, feature_in_gpu, kernel_in_gpu, 5, 3);
	//将GPU计算好的内存返回给CPU
	//copy back
	cudaMemcpy(feature_out_cpu, out_gpu, sizeof(float)*step1_out_size, cudaMemcpyDeviceToHost);

	////输出featrue_out_cpu 结果 调试用
	printf("---------------------------------------\n");
	printf("-------输出featrue_out_cpu结果-----------\n");
	printf("---------------------------------------\n");
	for(int kernel_num=0; kernel_num <10; kernel_num++){
		for (int layer = 0; layer < 3; layer++){
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 5; j++) {
					printf("%f\t", feature_out_cpu[(kernel_num*3+layer) * 5 * 5 + i * 5 + j]);
				}
			}
			printf(" \n             一层的数值        \n");
		}
		printf("\n---------------------------------\n");
	}

	printf("\n  --------- conv step 1 全部完成 ------ \n");

	cudaFree(feature_in_gpu);
	cudaFree(kernel_in_gpu);
	cudaFree(out_gpu);
	printf("\n  --------- 释放指针完成 ------ \n");

}

void test_conv_step1_new() {
	//确定我要开采的feature大小
	int arr_size_3D = 3 * 5 * 5;
	//kernel的大小
	int kernel_3d = 10 * 3 * 3 * 3;
	//step1_out大小
	int step1_out_size = 10 * 3 * 5 * 5;

	/*
	定义数据指针时 我们需要定义两种
	一个是CPU中的数据指针  一种是传到GPU中的数据指针
	为方便我们我们直接都定义一维数组
	*/

	// 首先是cpu中的 定义数组并赋值
	float *feature_in_cpu = (float*)malloc(arr_size_3D * sizeof(float));
	float *kernel_in_cpu = (float*)malloc(kernel_3d * sizeof(float));
	float *feature_out_cpu = (float*)malloc(step1_out_size * sizeof(float));

	////////////////////////////////////////////////////////////////////
	//-------------------  数据赋值部分 -----------------------
	//feature赋值
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				feature_in_cpu[layer * 5 * 5 + i * 5 + j] = layer + 1;
			}
		}
	}

	//kernel 赋值  4维循环
	for (int kernel_num = 0; kernel_num < 10; kernel_num++) {
		for (int layer = 0; layer < 3; layer++) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					kernel_in_cpu[(kernel_num * 3 + layer) * 3 * 3 + i * 3 + j] = kernel_num + 1;
				}
			}
		}
	}

	//-----------------------------数据赋值完成------------------------------
	//////////////////////////////////////////////////////////////////////////

	////输出featrue结果 调试用
	printf(" \n featrue初始化 \n ");
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				printf("%f\t", feature_in_cpu[layer * 5 * 5 + i * 5 + j]);
			}
		}
		printf("\n");
	}

	printf("--------------------------------------------------------");

	////输出kernel结果 调试用  4维
	printf(" \n kernel的初始化 \n ");
	for (int kernel_num = 0; kernel_num < 10; kernel_num++) {
		for (int layer = 0; layer < 3; layer++) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					printf("%f\t", kernel_in_cpu[(kernel_num * 3 + layer) * 3 * 3 + i * 3 + j]);
				}
			}
			printf("\n\n");
		}
		printf("------------- 第 %d 个卷积核数据 --------------\n", kernel_num);
	}

	printf("\n 初始化一切正常！！！\n");

	//定义GPU的数组指针 define GPU value pointer 
	float *feature_in_gpu, *kernel_in_gpu, *out_gpu;
	//他们都是要传到GPUcache中的数据

	// 在 GPU中 用cudaMalloc开辟内存空间
	//一定要记得传进去 二级指针
	cudaMalloc((void**)&feature_in_gpu, sizeof(float)*arr_size_3D);
	cudaMalloc((void**)&kernel_in_gpu, sizeof(float)*kernel_3d);
	cudaMalloc((void**)&out_gpu, sizeof(float)*step1_out_size);

	// 将CPU中的数据传到 GPU内存中
	cudaMemcpy(feature_in_gpu, feature_in_cpu, sizeof(float)*arr_size_3D, cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_in_gpu, kernel_in_cpu, sizeof(float)*kernel_3d, cudaMemcpyHostToDevice);

	printf("\n 内存COPY完成 \n");
	//输出结果 调试用

	//设计传入的线程  启用线程
	dim3 dimGrid(1,1,10);
	dim3 dimBlock(5,5);// 线程的形状
	
	//调用CUDA API时我们应该传入的是device的内存指针！！！
	//否则直接就调用不了kernel
	conv_step1_new<<<dimGrid, dimBlock >>>(out_gpu, feature_in_gpu, kernel_in_gpu, 5, 3,3);
	//将GPU计算好的内存返回给CPU
	//copy back
	cudaMemcpy(feature_out_cpu, out_gpu, sizeof(float)*step1_out_size, cudaMemcpyDeviceToHost);

	////输出featrue_out_cpu 结果 调试用
	printf("---------------------------------------\n");
	printf("-------输出featrue_out_cpu结果-----------\n");
	printf("---------------------------------------\n");
	for (int kernel_num = 0; kernel_num < 10; kernel_num++) {
		for (int layer = 0; layer < 3; layer++) {
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 5; j++) {
					printf("%f\t", feature_out_cpu[(kernel_num * 3 + layer) * 5 * 5 + i * 5 + j]);
				}
			}
			printf(" \n             一层的数值        \n");
		}
		printf("\n---------------------------------\n");
	}

	printf("\n  --------- conv step 1 全部完成 ------ \n");

	cudaFree(feature_in_gpu);
	cudaFree(kernel_in_gpu);
	cudaFree(out_gpu);
	printf("\n  --------- 释放指针完成 ------ \n");

}

void test_conv_step2() {
	//确定我要开采的feature大小
	int arr_size_3D = 9 * 5 * 5;   //3 个卷积核 *3层=9  

	//step2_out大小
	int step2_out_size =  3 * 5 * 5;

	/*
	定义数据指针时 我们需要定义两种
	一个是CPU中的数据指针  一种是传到GPU中的数据指针
	为方便我们直接都定义一维数组
	*/

	// 首先是cpu中的 定义数组并赋值
	float *feature_in_cpu = (float*)malloc(arr_size_3D * sizeof(float));
	float *feature_out_cpu = (float*)malloc(step2_out_size * sizeof(float));

	////////////////////////////////////////////////////////////////////
	//-------------------  数据赋值部分 -----------------------
	//feature赋值
	for (int kernel = 0; kernel < 3; kernel++){
		for (int layer = 0; layer < 3; layer++){
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 5; j++) {
					feature_in_cpu[kernel* 3 * 5 * 5+ layer*5*5 + i * 5 + j] = kernel + 1;
				}
			}
		}
	}


	//kernel 赋值  4维循环
	/*for (int kernel_num = 0; kernel_num < 10; kernel_num++) {
		for (int layer = 0; layer < 3; layer++) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					kernel_in_cpu[(kernel_num * 3 + layer) * 3 * 3 + i * 3 + j] = kernel_num + 1;
				}
			}
		}
	}*/

	//-----------------------------数据赋值完成------------------------------
	//////////////////////////////////////////////////////////////////////////


	////输出featrue结果 调试用
	printf(" \n featrue_step1 初始化 \n ");
	for (int kernel = 0; kernel < 3; kernel++) {
		for (int layer = 0; layer < 3; layer++) {
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 5; j++) {
					printf("%f\t", feature_in_cpu[kernel * 3 * 5 * 5 + layer * 5 * 5 + i * 5 + j]);
				}
			}
			printf("\n");
		}
		printf("------------- 第 %d 个卷积核数据 --------------\n", kernel);
	}
	printf("\n------------------------feature_out 初始化完成--------------------------------\n");
	printf("\n 初始化一切正常！！！\n");
	//-------------------------------------------------------------------------------------------------

	//定义GPU的数组指针 define GPU value pointer
	float *feature_in_gpu, *out_gpu;
	//他们都是要传到GPUcache中的数据

	// 在 GPU中 用cudaMalloc开辟内存空间
	//一定要记得传进去 二级指针
	cudaMalloc((void**)&feature_in_gpu, sizeof(float)*arr_size_3D);
	cudaMalloc((void**)&out_gpu, sizeof(float)*step2_out_size);

	// 将CPU中的数据传到 GPU内存中
	cudaMemcpy(feature_in_gpu, feature_in_cpu, sizeof(float)*arr_size_3D, cudaMemcpyHostToDevice);

	printf("\n 内存COPY完成 \n");

	////////////////////////////////////////////////////////////////////////////////////////////
	/////////      前期数据准备好  接下来设计线程结构调用kernel函数                     ////////
	///////////////////////////////////////////////////////////////////////////////////////////
	//设计传入的线程  启用线程
	dim3 dimBlock(5, 5);// 线程的形状
	dim3 dimGrid(3);

	//调用CUDA API时我们应该传入的是device的内存指针！！！
	//否则直接就调用不了kernel
	conv_step2<<<dimGrid, dimBlock>>>(out_gpu, feature_in_gpu, 3,3,5);
	
	//将GPU计算好的内存返回给CPU
	//copy back from gpu
	cudaMemcpy(feature_out_cpu, out_gpu, sizeof(float)*step2_out_size, cudaMemcpyDeviceToHost);

	////// 输出featrue_out_cpu 结果 调试用   //////////
	printf("---------------------------------------\n");
	printf("-------输出featrue_out_cpu结果-----------\n");
	printf("---------------------------------------\n");
	//for (int kernel_num = 0; kernel_num < 10; kernel_num++) {
		for (int layer = 0; layer < 3; layer++) {
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 5; j++) {
					printf("%f\t", feature_out_cpu[layer * 5 * 5 + i * 5 + j]);
				}
			}
			printf(" \n             一层的数值        \n");
		}
		printf("\n---------------------------------\n");
	//}

	printf("\n  --------- conv step 2 全部完成 ------ \n");

	cudaFree(feature_in_gpu);
	cudaFree(out_gpu);
	printf("\n  --------- 释放指针完成 ------ \n");

}

void test_conv_step2_new() {
	//确定我要开采的feature大小
	int arr_size_3D = 3* 3 * 64 * 64;   //3 个卷积核 *3层=9  

	//step2_out大小
	int step2_out_size = 3 * 64 * 64;

	/*
	定义数据指针时 我们需要定义两种
	一个是CPU中的数据指针  一种是传到GPU中的数据指针
	为方便我们直接都定义一维数组
	*/

	// 首先是cpu中的 定义数组并赋值
	float *feature_in_cpu = (float*)malloc(arr_size_3D * sizeof(float));
	float *feature_out_cpu = (float*)malloc(step2_out_size * sizeof(float));

	////////////////////////////////////////////////////////////////////
	//-------------------  数据赋值部分 -----------------------
	//feature赋值
	for (int kernel = 0; kernel < 3; kernel++) {
		for (int layer = 0; layer < 3; layer++) {
			for (int i = 0; i < 64; i++) {
				for (int j = 0; j < 64; j++) {
					feature_in_cpu[kernel * 3 * 64 * 64 + layer * 64 * 64 + i * 64 + j] = kernel + 1;
				}
			}
		}
	}

	//-----------------------------数据赋值完成------------------------------
	//////////////////////////////////////////////////////////////////////////


	////输出featrue结果 调试用
	printf(" \n featrue_step1 初始化 \n ");
	for (int kernel = 0; kernel < 3; kernel++) {
		for (int layer = 0; layer < 3; layer++) {
			for (int i = 0; i < 64; i++) {
				for (int j = 0; j < 64; j++) {
					printf("%f\t", feature_in_cpu[kernel * 3 * 64 * 64 + layer * 64 * 64 + i * 64 + j]);
				}
			}
			printf("\n");
		}
		printf("------------- 第 %d 个卷积核数据 --------------\n", kernel);
	}
	printf("\n------------------------feature_out 初始化完成--------------------------------\n");
	printf("\n 初始化一切正常！！！\n");
	//-------------------------------------------------------------------------------------------------

	//定义GPU的数组指针 define GPU value pointer
	float *feature_in_gpu, *out_gpu;
	//他们都是要传到GPUcache中的数据

	// 在 GPU中 用cudaMalloc开辟内存空间
	//一定要记得传进去 二级指针
	cudaMalloc((void**)&feature_in_gpu, sizeof(float)*arr_size_3D);
	cudaMalloc((void**)&out_gpu, sizeof(float)*step2_out_size);

	// 将CPU中的数据传到 GPU内存中
	cudaMemcpy(feature_in_gpu, feature_in_cpu, sizeof(float)*arr_size_3D, cudaMemcpyHostToDevice);

	printf("\n 内存COPY完成 \n");

	////////////////////////////////////////////////////////////////////////////////////////////
	/////////      前期数据准备好  接下来设计线程结构调用kernel函数                     ////////
	///////////////////////////////////////////////////////////////////////////////////////////
	//设计传入的线程  启用线程
	dim3 dimGrid(2,2,3);
	dim3 dimBlock(32, 32);// 线程的形状

	//调用CUDA API时我们应该传入的是device的内存指针！！！
	//否则直接就调用不了kernel
	conv_step2_new<<<dimGrid, dimBlock >> > (out_gpu, feature_in_gpu, 3, 64);

	//将GPU计算好的内存返回给CPU
	//copy back from gpu
	cudaMemcpy(feature_out_cpu, out_gpu, sizeof(float)*step2_out_size, cudaMemcpyDeviceToHost);

	////// 输出featrue_out_cpu 结果 调试用   //////////
	printf("---------------------------------------\n");
	printf("-------输出featrue_out_cpu结果-----------\n");
	printf("---------------------------------------\n");
	//for (int kernel_num = 0; kernel_num < 10; kernel_num++) {
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 64; i++) {
			for (int j = 0; j < 64; j++) {
				printf("%f\t", feature_out_cpu[layer * 64 * 64 + i * 64 + j]);
			}
		}
		printf(" \n             一层的数值        \n");
	}
	printf("\n---------------------------------\n");
	//}

	printf("\n  --------- conv step 2 全部完成 ------ \n");

	cudaFree(feature_in_gpu);
	cudaFree(out_gpu);
	printf("\n  --------- 释放指针完成 ------ \n");
}

void test_pool() {
	//确定我要开采的feature大小
	int arr_size_3D = 3 * 8 * 8;   //3层 每层的feature_map为 8*8大小  

	//step2_out大小
	int pool_out_size = 3 * 4 * 4;

	/*
	定义数据指针时 我们需要定义两种
	一个是CPU中的数据指针  一种是传到GPU中的数据指针
	为方便我们直接都定义一维数组
	*/

	// 首先是cpu中的 定义数组并赋值
	float *feature_in_cpu = (float*)malloc(arr_size_3D * sizeof(float));
	float *pool_out_cpu = (float*)malloc(pool_out_size * sizeof(float));

	////////////////////////////////////////////////////////////////////
	//-------------------  数据赋值部分 -----------------------
	//feature赋值
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++){
				feature_in_cpu[layer * 8 * 8 + i *8 + j] = 1.0*(i+j);
			}
		}
	}

	//-----------------------------数据赋值完成------------------------------
	//////////////////////////////////////////////////////////////////////////

	////输出featrue结果 调试用
	printf(" \n featrue_in 初始化 \n ");
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				printf("%f\t",feature_in_cpu[layer * 8 * 8 + i * 8 + j]);
				}
			printf("\n");
		}
		printf("------------- 第 %d 层 --------------\n", layer);
	}
	printf("\n------------------------feature_in 初始化完成--------------------------------\n");
	printf("\n 初始化一切正常！！！\n");
	//-------------------------------------------------------------------------------------------------

	//定义GPU的数组指针 define GPU value pointer
	float *feature_in_gpu, *out_gpu;
	//他们都是要传到GPUcache中的数据

	//// 在 GPU中 用cudaMalloc开辟内存空间
	////一定要记得传进去 二级指针
	cudaMalloc((void**)&feature_in_gpu, sizeof(float)*arr_size_3D);
	cudaMalloc((void**)&out_gpu, sizeof(float)*pool_out_size);

	// 将CPU中的数据传到 GPU内存中
	cudaMemcpy(feature_in_gpu, feature_in_cpu, sizeof(float)*arr_size_3D, cudaMemcpyHostToDevice);
	cudaError_t result = cudaMemcpy(feature_in_gpu, feature_in_cpu, sizeof(float)*arr_size_3D, cudaMemcpyHostToDevice);

	printf("\n #########内存COPY完成  %d ########## \n",result);

	////////////////////////////////////////////////////////////////////////////////////////////
	/////////      前期数据准备好  接下来设计线程结构调用kernel函数                     ////////
	///////////////////////////////////////////////////////////////////////////////////////////
	//设计传入的线程  启用线程
	dim3 dimBlock(4, 4);// 线程的形状
	dim3 dimGrid(3);

	////调用CUDA API时我们应该传入的是device的内存指针！！！
	////否则直接就调用不了kernel
	pool<<<dimGrid, dimBlock>>> (out_gpu, feature_in_gpu, 8, 2);

	//将GPU计算好的内存返回给CPU
	//copy back from gpu
	cudaMemcpy(pool_out_cpu, out_gpu, sizeof(float)*pool_out_size, cudaMemcpyDeviceToHost);

	//////// 输出featrue_out_cpu 结果 调试用   //////////
	printf("---------------------------------------\n");
	printf("-------输出featrue_out_cpu结果-----------\n");
	printf("---------------------------------------\n");
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				printf("%f\t", pool_out_cpu[layer * 4 * 4 + i * 4 + j]);
			}
		}
		printf(" \n             一层的数值        \n");
	}
	printf("\n---------------------------------\n");
	//}

	cudaFree(feature_in_gpu);
	cudaFree(out_gpu);
	printf("\n  --------- 释放指针完成 ------ \n");

}

void test_pool_new() {
	//确定我要开采的feature大小
	int arr_size_3D = 3 * 8 * 8;   //3层 每层的feature_map为 8*8大小  

	//step2_out大小
	int pool_out_size = 3 * 4 * 4;

	/*
	定义数据指针时 我们需要定义两种
	一个是CPU中的数据指针  一种是传到GPU中的数据指针
	为方便我们直接都定义一维数组
	*/

	// 首先是cpu中的 定义数组并赋值
	float *feature_in_cpu = (float*)malloc(arr_size_3D * sizeof(float));
	float *pool_out_cpu = (float*)malloc(pool_out_size * sizeof(float));

	////////////////////////////////////////////////////////////////////
	//-------------------  数据赋值部分 -----------------------
	//feature赋值
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				feature_in_cpu[layer * 8 * 8 + i * 8 + j] = i + j;
			}
		}
	}

	//-----------------------------数据赋值完成------------------------------
	//////////////////////////////////////////////////////////////////////////

	////输出featrue结果 调试用
	printf(" \n featrue_in 初始化 \n ");
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				printf("%f\t", feature_in_cpu[layer * 8 * 8 + i * 8 + j]);
			}
			printf("\n");
		}
		printf("------------- 第 %d 层 --------------\n", layer);
	}
	printf("\n------------------------feature_in 初始化完成--------------------------------\n");
	printf("\n 初始化一切正常！！！\n");
	//-------------------------------------------------------------------------------------------------

	//定义GPU的数组指针 define GPU value pointer
	float *feature_in_gpu, *out_gpu;
	//他们都是要传到GPUcache中的数据

	//// 在 GPU中 用cudaMalloc开辟内存空间
	////一定要记得传进去 二级指针
	cudaMalloc((void**)&feature_in_gpu, sizeof(float)*arr_size_3D);
	cudaMalloc((void**)&out_gpu, sizeof(float)*pool_out_size);

	// 将CPU中的数据传到 GPU内存中
	cudaMemcpy(feature_in_gpu, feature_in_cpu, sizeof(float)*arr_size_3D, cudaMemcpyHostToDevice);

	//printf("\n 内存COPY完成 \n");

	////////////////////////////////////////////////////////////////////////////////////////////
	/////////      前期数据准备好  接下来设计线程结构调用kernel函数                     ////////
	///////////////////////////////////////////////////////////////////////////////////////////
	//设计传入的线程  启用线程
	dim3 dimBlock(4, 4);// 线程的形状
	dim3 dimGrid(1,1,3);

	////调用CUDA API时我们应该传入的是device的内存指针！！！
	////否则直接就调用不了kernel
	pool_new<<<dimGrid, dimBlock>>> (out_gpu, feature_in_gpu, 8, 2);

	//将GPU计算好的内存返回给CPU
	//copy back from gpu
	cudaMemcpy(pool_out_cpu, out_gpu, sizeof(float)*pool_out_size, cudaMemcpyDeviceToHost);

	//////// 输出featrue_out_cpu 结果 调试用   //////////
	printf("---------------------------------------\n");
	printf("-------输出featrue_out_cpu结果-----------\n");
	printf("---------------------------------------\n");
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				printf("%f\t", pool_out_cpu[layer * 4 * 4 + i * 4 + j]);
			}
		}
		printf(" \n             一层的数值        \n");
	}
	printf("\n---------------------------------\n");
	//}

	cudaFree(feature_in_gpu);
	cudaFree(out_gpu);
	printf("\n  --------- 释放指针完成 ------ \n");

}

void test_FC_SharedMem() {
	//确定我要开采的feature大小
	int arr_size_3D = 128 ;
	//kernel的大小
	int weight_size = 128*256 ; 
	//step1_out大小
	int featureout_size = 256 ;

	float *feature_in_cpu = (float*)malloc(arr_size_3D * sizeof(float));
	float *weight_in_cpu = (float*)malloc(weight_size * sizeof(float));
	float *feature_out_cpu = (float*)malloc(featureout_size * sizeof(float));

	////////////////////////////////////////////////////////////////////
	//-------------------  数据赋值部分 -----------------------
	//feature赋值
	for(int i=0; i < arr_size_3D ;i++){
		feature_in_cpu[i] = i+1;
	}

	//kernel 赋值  
	for(int raw=0;raw<256;raw++){
		for(int col=0;col<128;col++){
			weight_in_cpu[raw*128+col] = raw;
		}
	}

	//输出调试结果
	for (int i = 0; i < arr_size_3D; i++) {
		printf("%f\t",feature_in_cpu[i]);
	}
	printf("\n     feature_test \n");

	for (int raw = 0; raw < 256; raw++) {
		for (int col = 0; col < 128; col++) {
			printf("%f\t", weight_in_cpu[raw * 128 + col]);
		}
		printf("\n \n");
	}
	printf("\n     weight test \n");

	//定义GPU的数组指针 define GPU value pointer 
	float *feature_in_gpu, *weight_in_gpu, *out_gpu;
	//他们都是要传到GPU globalMEm中的数据

	// 在 GPU中 用cudaMalloc开辟内存空间
	//一定要记得传进去 二级指针
	cudaMalloc((void**)&feature_in_gpu, sizeof(float)*arr_size_3D);
	cudaMalloc((void**)&weight_in_gpu, sizeof(float)*weight_size);
	cudaMalloc((void**)&out_gpu, sizeof(float)*featureout_size);

	// 将CPU中的数据传到 GPU内存中
	cudaMemcpy(feature_in_gpu, feature_in_cpu, sizeof(float)*arr_size_3D, cudaMemcpyHostToDevice);
	cudaMemcpy(weight_in_gpu, weight_in_cpu, sizeof(float)*weight_size, cudaMemcpyHostToDevice);

	//设计传入的线程  启用线程
	dim3 dimGrid(16);// 线程的形状
	dim3 dimBlock(16);

	FC_SharedMem<<<dimGrid, dimBlock>>>(feature_in_gpu, weight_in_gpu, out_gpu, 128);

	cudaMemcpy(feature_out_cpu, out_gpu, sizeof(float)*featureout_size, cudaMemcpyDeviceToHost);

	////输出featrue_out_cpu 结果 调试用
	printf("---------------------------------------\n");
	printf("-------输出featrue_out_cpu结果-----------\n");
	printf("---------------------------------------\n");
	for(int i=0; i <featureout_size; i++){
		if (i % 32 == 0)  printf("\n");
		printf("%f\t", feature_out_cpu[i]);
	}
	cudaFree(feature_in_gpu);
	cudaFree(weight_in_gpu);
	cudaFree(out_gpu);
}

//完整的测试案例工程
void FullCnnProject() {

	//opencv读取图片 512*512
	Mat raw_img = imread("D:\\asus\\Desktop\\cudaProject\\cudaProject\\Lena.jpg");
	Mat grey_img,small_img;
	printf("图片大小为 %d行  %d列  %d通道 \n ", raw_img.rows, raw_img.cols, raw_img.channels());
	cout << "像素类型为："<< raw_img.type() << endl;
	//imshow("读取图像展示", img);
	//waitKey(1500);
	//转换成灰度图
	//cvtColor(raw_img, grey_img, CV_BGR2GRAY);
	//printf("图片大小为 %d行  %d列  %d通道 \n ", grey_img.rows, grey_img.cols, grey_img.channels());

	resize(raw_img, small_img, Size(64, 64), INTER_AREA);
	printf("图片缩放后的大小为 %d行  %d列  %d通道 \n ", small_img.rows, small_img.cols, small_img.channels());

	//将图像像素数值转换成浮点型
	//我们自己定义float型数组，然后将数据转换过来就行
	int pic_input_size = small_img.rows * small_img.cols*raw_img.channels();

	float * img_input=(float *)malloc(pic_input_size *sizeof(float));
	//循环 assignment
	for (int channel = 0; channel < small_img.channels(); channel++)
	{
		for (int i = 0; i < small_img.rows; i++)
		{
			for (int j = 0; j < small_img.cols; j++)
			{
				img_input[channel*small_img.rows * small_img.cols+i*small_img.cols + j] =
					(float)small_img.data[channel*small_img.rows * small_img.cols+i*small_img.cols + j] / 256;
			}
		}
	}

	//imshow("缩小图像展示", small_img);
	//waitKey(1500);

	for (int channel = 0; channel < small_img.channels(); channel++)
	{
		for (int i = 0; i < small_img.rows; i++)
		{
			for (int j = 0; j < small_img.cols; j++)
			{
				printf("%f \t", img_input[channel*small_img.rows * small_img.cols + i * small_img.cols + j]);
			}
			printf(" \n \n");
		}
		printf(" \n  图像一层  \n");
	}
	cout << "##########################以上为图像数组输出########################### " << "\n"<< "\n";

	//////////////////////////////////////////////////////////////////////
	///////        接下来就是传入整个卷积网络的运算了！！！      /////////
	//////////////////////////////////////////////////////////////////////
	
	/////////////////////  一 卷积 step1 ////////////////////////////////////
	//首先 原图像数组已经有了
	int input_3d_size =small_img.channels()* small_img.rows*small_img.cols;
	int kernel_3d_size = 10 * small_img.channels() * 3*3;
	//step1_out大小
	int step1_out_size = 10 * small_img.channels() * small_img.rows*small_img.cols;

	float *kernel_in_cpu = (float*)malloc(kernel_3d_size * sizeof(float));
	float *feature_out_cpu = (float*)malloc(step1_out_size * sizeof(float));
	//  10个 kernel 赋值
	for(int num=0;num<10;num++){
		for (int  layer = 0; layer < small_img.channels(); layer++){
			for(int i = 0; i < Kernel_size; i++) {
				for (int j = 0; j < Kernel_size; j++) {
					kernel_in_cpu[num * Kernel_size * Kernel_size*small_img.channels()
									+ layer* Kernel_size * Kernel_size
									+ i * Kernel_size + j] = i + 1;
				}
			}
		}
	}

	//输出结果调试
	for (int num = 0; num < 10; num++) {
		for (int layer = 0; layer < small_img.channels(); layer++) {
			for (int i = 0; i < Kernel_size; i++) {
				for (int j = 0; j < Kernel_size; j++) {
					printf("%f \t", kernel_in_cpu[num * Kernel_size * Kernel_size*small_img.channels()
						+ layer * Kernel_size * Kernel_size
						+ i * Kernel_size + j] );
				}
			}
		}
		printf(" \n  一个 卷积核的数据  \n ");
	}

	float *feature_in_gpu, *kernel_in_gpu, *out_gpu;
	cudaMalloc((void**)&feature_in_gpu, sizeof(float)*input_3d_size);
	cudaMalloc((void**)&kernel_in_gpu, sizeof(float)*kernel_3d_size);
	cudaMalloc((void**)&out_gpu, sizeof(float)*step1_out_size);

	printf("\n cudaMalloc 完成\n");

	// 将CPU中的数据传到 GPU内存中
	cudaMemcpy(feature_in_gpu, img_input, sizeof(float)*input_3d_size, cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_in_gpu, kernel_in_cpu, sizeof(float)*kernel_3d_size, cudaMemcpyHostToDevice);

	printf("\n cudaMemcpy 完成\n");

	//设计传入的线程  启用线程
	dim3 dimGrid(2,2,10);
	dim3 dimBlock(32,32);// 线程的形状

	//调用CUDA API时我们应该传入的是device的内存指针！！！
	//否则直接就调用不了kernel
	conv_step1_new<<<dimGrid, dimBlock >>>(out_gpu, feature_in_gpu, kernel_in_gpu,64,3,3);
	printf("\n函数调用完成\n");
	//将GPU计算好的内存返回给CPU
	//copy back
	cudaMemcpy(feature_out_cpu, out_gpu, sizeof(float)*step1_out_size, cudaMemcpyDeviceToHost);

	//输出featrue_out_cpu 结果 调试用
	printf("---------------------------------------\n");
	printf("-------输出featrue_out_cpu结果-----------\n");
	printf("---------------------------------------\n");
	/*for (int num = 0; num < 10; num++) {
		for (int layer = 0; layer < small_img.channels(); layer++) {
			for (int i = 0; i < small_img.rows; i++) {
				for (int j = 0; j < small_img.cols; j++) {
					printf("%f \t", feature_out_cpu[num * small_img.rows* small_img.cols*small_img.channels()
						+ layer * small_img.rows* small_img.cols
						+ i * small_img.cols + j]);
				}
			}
			printf(" \n  一个层卷积层的结果  \n ");
		}
	}*/
	printf("\n  --------- conv step 1 全部完成 ------ \n");

	cudaFree(feature_in_gpu);
	cudaFree(kernel_in_gpu);

	//out_gpu不释放继续往下一个函数传值

	////////////////////////////////////////////////////////////////////////////////////////
	///////           conv step 2       我们保留了之前的  gpu_out指针            //////////////
	////////////////////////////////////////////////////////////////////////////////////////
	int conv_setp2_out_size = small_img.rows*small_img.cols * 10;

	float * conv_step2_out_cpu, *conv_step2_out_gpu;

	conv_step2_out_cpu = (float*)malloc(conv_setp2_out_size * sizeof(float));
	cudaMalloc((void**)&conv_step2_out_gpu, sizeof(float)*conv_setp2_out_size);

	dim3 gridDim(2,2,10);
	dim3 blockDim(32,32);

	conv_step2_new<<<gridDim , blockDim>>>(conv_step2_out_gpu, out_gpu, 3, 64);
	cudaMemcpy(conv_step2_out_cpu, conv_step2_out_gpu, sizeof(float)*conv_setp2_out_size, cudaMemcpyDeviceToHost);

	printf("---------------------------------------\n");
	printf("-------输出conv_step2_cpu结果-----------\n");
	printf("---------------------------------------\n");
	for (int num = 0; num < 10; num++) {
		for (int layer = 0; layer < 1; layer++) {
			for (int i = 0; i < small_img.rows; i++) {
				for (int j = 0; j < small_img.cols; j++) {
					printf("%f \t", feature_out_cpu[num * small_img.rows* small_img.cols * 1
						+ layer * small_img.rows* small_img.cols
						+ i * small_img.cols + j]);
				}
			}
			printf(" \n  第%d个层卷积层的结果  \n ", num);
		}
	}
	//在这里可以释放上一步的输出的gpu指针


	////////////////////////////////////////////////////////////////////
	////////////////////   池化 操作   conv_step2_out_gpu ////////////////////////////////
	////////////////////////////////////////////////////////////////////
	//池化操作没有 权重矩阵的输入
	int pool_out_size = 8 * 8 * 10;

	float * pool_out_cpu, *pool_out_gpu;

	//开辟内存空间
	cudaMalloc((void**)&pool_out_gpu, sizeof(float)*pool_out_size);
	pool_out_cpu = (float*)malloc(pool_out_size * sizeof(float));

	//kernel函数调用
	dim3 gridDim_pool(1, 1, 10);
	dim3 gridBlock_pool(8, 8);
	pool_new<<<gridDim_pool, gridBlock_pool >>>(pool_out_gpu, conv_step2_out_gpu, 64, 8);

	//结果拷回cpu
	cudaMemcpy(pool_out_cpu, pool_out_gpu, sizeof(float)*pool_out_size, cudaMemcpyDeviceToHost);

	for (int num = 0; num < 10; num++) {
		for (int layer = 0; layer < 1; layer++) {
			for (int i = 0; i < 8; i++) {
				for (int j = 0; j < 8; j++) {
					printf("%f \t", pool_out_cpu[num * 64 * 1
						+ i * 8 + j]);
				}
			}
			printf(" \n  第%d个层的结果  \n ", num);
		}
	}
	printf(" \n  池化操作运行成功！！！NB  \n ");

	// 释放可以释放掉的指针
	cudaFree(conv_step2_out_gpu);

	////////////////////////////////////////////////////////////////////////
	/////////////          全连接层(矩阵运算)      /////////////////////////
	////////////////////////////////////////////////////////////////////////
	//定义开辟数据空间的大小和一些宏变量
	int weight_size = 8 * 8 * 10 * 10;
	int fc_out_size = 10;

	float * fc_out_cpu, *fc_out_gpu,* weight_in_cpu,*weight_in_gpu;

	//开辟对应的空间
	fc_out_cpu = (float *)malloc(fc_out_size * sizeof(float));
	weight_in_cpu = (float *)malloc(weight_size * sizeof(float));
	cudaMalloc((void**)&fc_out_gpu, sizeof(float)*fc_out_size);
	cudaMalloc((void**)&weight_in_gpu, sizeof(float)*weight_size);

	//给权重数组赋初值 一定要在 mencopy之前！
	for (int raw = 0; raw < 10; raw++) {
		for (int i = 0; i < 8 * 8 * 10; i++)
		{
			weight_in_cpu[raw * 8 * 8 * 10 + i] = raw *0.1+1;
		}
	}

	for (int raw = 0; raw < 10; raw++) {
		for (int i = 0; i < 8 * 8 * 10; i++)
		{
			printf(" %f \t",weight_in_cpu[raw * 8 * 8 * 10 + i]);
		}
		printf("\n");
		printf("#########################################");
		printf("########### 一个数据 ############\n\n");
	}

	cudaMemcpy(weight_in_gpu, weight_in_cpu, sizeof(float)*weight_size, cudaMemcpyHostToDevice);

	//设计传入 线程结构，调用kernel函数
	dim3 dimGrid_fc(10);
	dim3 dimBlock_fc(1);
	FC_SharedMem <<<dimGrid_fc, dimBlock_fc>>> (pool_out_gpu, weight_in_gpu, fc_out_gpu, 8 * 8 * 10);

	//结果拷贝回cpu
	cudaMemcpy(fc_out_cpu, fc_out_gpu, sizeof(float)*fc_out_size, cudaMemcpyDeviceToHost);

	int best_index=-1;
	float temp = 0.0;
	for (int i = 0; i < fc_out_size; i++)
	{
		printf("预测第 %d 个类别的 得分是 %f \n", i, fc_out_cpu[i]);
		if (fc_out_cpu[i] > temp) {
			temp = fc_out_cpu[i];
			best_index = i;
		}
	}
	printf("\n最后预测的是第%d类!!!!\n", best_index);


	cudaFree(weight_in_gpu);
	cudaFree(fc_out_gpu);
	free(weight_in_cpu);
	free(fc_out_cpu);

}

int main(void){
	FullCnnProject();
}