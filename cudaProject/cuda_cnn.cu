#include "cuda_cnn.cuh"
#include"device_functions.h"

//  conv_step1实现   只是把每层的卷积单层卷积对应起来
/*
float * step1_out   step1_out的featuremap指针
float * feature_in   传入的featuremap指针
int con_core_in     卷积核的指针
int featuremap_size  featuremap尺寸  in和out的尺寸都一样
int coresize         卷积核的尺寸
int layers_in_featuremap   每个feature有几层
*/
__global__ void conv_step1(float *step1_out, float * feature_in,
							float * con_core_in, int featuremap_size, int coresize)
{
	// 这是取的数量的总个数
	int layer_nums = gridDim.y;  // block的第二维为卷积核也是feature的层数
	
	// 定位到线程所在的block
	int kernel_now = blockIdx.x;  //   目前是第几个kernel
	int layer_now =  blockIdx.y;  //   目前在第几层中

	// 定位到block中的线程,也就是kernel函数真实地执行者
	int row_now = threadIdx.x;  // 表示所在层的行
	int col_now =  threadIdx.y;  // 表示所在层的列

	//每个线程他具体做了什么呢？
	//他把对应到的feature数据和对应的卷积核相乘
	// 是一个3x3的小循环
	//我们现在的要做的是定位到featrue中的对应的数据
	
	//输出的out数组的定位
	int out_position = (kernel_now*layer_nums + layer_now)*(featuremap_size*featuremap_size) + row_now * featuremap_size + col_now; //输出结果的定位
	
	// feature中的数据定位，定位到层数即可，最后每次再取
	int feature_layer_start= layer_now * (featuremap_size*featuremap_size);

	//  kernel定位到个数和层数
	int kernel_pos_start = (kernel_now*layer_nums+layer_now)*coresize*coresize;

	//////////////////////////////////////////////////////////////
	////         开始二层循环计算一个卷积的值                 ////
	//////////////////////////////////////////////////////////////
	float temp=0.0f;
	
	//因为一个卷积运算是从最左上的边角开始的  我们要定位到他的卷积开始的行和列
	int starti = row_now - coresize / 2;
	int startj = col_now - coresize / 2;

	for (int i = starti; i < starti + coresize; i++){
		for (int j = startj; j < startj + coresize; j++){
			if (i >= 0 && j >= 0 && i < featuremap_size && j < featuremap_size)
			{
				temp =temp+con_core_in[kernel_pos_start+(i - starti)*coresize+(j - startj)] * feature_in[feature_layer_start+(i * featuremap_size)+j];
				//printf("GPU\n");
				//printf(" con_core_in[%d %d] = %f feature_in[%d %d] =%f\n", i, j, con_core_in[kernel_pos_start + (i - starti)*coresize + (j - startj)], i - starti, j - startj, feature_in[feature_layer_start + (i * featuremap_size) + j]);
			}
			
		}
	}
	//printf("\n temp= %f \n", temp);
	step1_out[out_position] = temp;
}

__global__ void conv_step1_new(float *step1_out, float * feature_in,
	float * con_core_in, int featuremap_size, int coresize,int layers_in_featuremap)
{
	//先求出一些宏观总体量
	int threadraws_inablock =blockDim.x;
	int threadcols_inablock = blockDim.y;
	
	int blockraws_inagrid = gridDim.x;
	int blockcols_inagrid = gridDim.y;

	//一层总共的数据数偏移量
	int offset_a_layer = (blockraws_inagrid* blockcols_inagrid)*(threadraws_inablock* threadcols_inablock);
	int offset_a_raw = blockcols_inagrid*threadcols_inablock;

	// 目前在第几个kernel中
	int kernel_now = blockIdx.z;   //

	// 定位到线程所在的block
	int block_x = blockIdx.x;  // 
	int block_y = blockIdx.y;   //  

	// 定位到block中的线程,也就是kernel函数真实地执行者
	int row_now_inablock = threadIdx.x;  // 表示所在层的行
	int col_now_inablock = threadIdx.y;  // 表示所在层的列

	//因为每个线程要计算N个数，(N与层数对应)
	for (int layer_now = 0; layer_now < layers_in_featuremap; layer_now++)
	{
		// 输出的out数组的定位
		int out_position = kernel_now * layers_in_featuremap * offset_a_layer
										+ layer_now * offset_a_layer
										+ (block_x * threadraws_inablock+ row_now_inablock)*offset_a_raw
										+ block_y * threadcols_inablock+col_now_inablock;
		//printf("输出位置%d个数据 \n",out_position);

		// feature中的数据定位，定位到层数即可，最后每次再取
		int feature_layer_start = layer_now * (featuremap_size*featuremap_size);

		//  kernel定位到个数和层数
		int kernel_pos_start = (kernel_now*layers_in_featuremap + layer_now)*coresize*coresize;

		//////////////////////////////////////////////////////////////
		////         开始二层循环计算一个卷积的值                 ////
		//////////////////////////////////////////////////////////////
		float temp = 0.0f;

		//因为一个卷积运算是从最左上的边角开始的  我们要定位到他的卷积开始的行和列
		int starti = (block_x * threadraws_inablock + row_now_inablock) - coresize / 2;
		int startj = (block_y * threadcols_inablock + col_now_inablock) - coresize / 2;

		for (int i = starti; i < starti + coresize; i++) {
			for (int j = startj; j < startj + coresize; j++) {
				if (i >= 0 && j >= 0 && i < featuremap_size && j < featuremap_size)
				{
					temp = temp + con_core_in[kernel_pos_start + (i - starti)*coresize + (j - startj)] * feature_in[feature_layer_start + (i * featuremap_size) + j];
				}
			}
		}
		//printf("\n temp= %f \n", temp);
		step1_out[out_position] = temp;
	}

}

//第二步是做累加，与卷积核无关了 只是纵向的累加
/*
float * feature_out   输出的feature指针
float * step1_out    conv_step1的结果矩阵
int core_num     卷积核个数   对应输出的featuremap的层数
int core_layers  卷积核层数   对应于每次累加我要跳多少个间隔来计算总的累加。
*/
__global__ void conv_step2(float * feature_out, float * step1_out,
				int core_num, int core_layers,int step1_out_featuremap_size) {
	// 定位到 输出层的 层号
	int out_layer_now = blockIdx.x;

	//定位到输出层的一层中的 x和y;
	int row_now = threadIdx.x;  // 表示所在层的行
	int col_now = threadIdx.y;  // 表示所在层的列

	//找到每次累加计算开始的地方
	int start_pos=out_layer_now * core_layers * (step1_out_featuremap_size*step1_out_featuremap_size)
						+ row_now*step1_out_featuremap_size+col_now;
	
	//求得输出的位置
	int out_pos=out_layer_now*step1_out_featuremap_size*step1_out_featuremap_size+ row_now*step1_out_featuremap_size+col_now;

	float temp=0.0f;

	/////////////////// 正式循环计算部分  ///////////////////
	//用循环开始累加运算 即计算一个输出值
	for(int i=0;i<core_layers;i++){
		temp= temp + step1_out[start_pos+i*step1_out_featuremap_size*step1_out_featuremap_size];
	}
	///////////////////////////////////////////////////////

	feature_out[out_pos] = temp;
}

//第二步是做累加，与卷积核无关了 只是纵向的累加
/*
float * feature_out   输出的feature指针
float * step1_out    conv_step1的结果矩阵
int core_num     卷积核个数   对应输出的featuremap的层数
int core_layers  卷积核层数   对应于每次累加我要跳多少个间隔来计算总的累加。
*/
__global__ void conv_step2_new(float * feature_out, float * step1_out,
	 int core_layers, int step1_out_featuremap_size) {
	
	int threadraws_inablock = blockDim.x;
	int threadcols_inablock = blockDim.y;

	int blockraws_inagrid = gridDim.x;
	int blockcols_inagrid = gridDim.y;
	// 定位到 输出层的 层号
	int out_layer_now = blockIdx.z;

	//定位到所在的block位置
	int block_raw_now = blockIdx.x;
	int block_col_now = blockIdx.y;

	int threadx_inbloack = threadIdx.x;
	int thready_inbloack = threadIdx.y;

	//定位到输出层的一层中的 x和y;
	int row_now = block_raw_now * threadraws_inablock+ threadx_inbloack;  // 表示所在层的行
	int col_now = block_col_now * threadcols_inablock+ thready_inbloack;  // 表示所在层的列

	//找到每次累加计算开始的地方
	int start_pos = out_layer_now * core_layers * (step1_out_featuremap_size*step1_out_featuremap_size)
		+ row_now * step1_out_featuremap_size + col_now;

	//求得输出的位置
	int out_pos = out_layer_now * step1_out_featuremap_size*step1_out_featuremap_size + row_now * step1_out_featuremap_size + col_now;

	float temp = 0.0f;

	/////////////////// 正式循环计算部分  ///////////////////
	//用循环开始累加运算 即计算一个输出值
	for (int i = 0; i < core_layers; i++) {
		temp = temp + step1_out[start_pos + i * step1_out_featuremap_size*step1_out_featuremap_size];
	}
	///////////////////////////////////////////////////////

	feature_out[out_pos] = temp;
}

//第一步和第二步合起来是一次完整的	卷积操作

__global__ void pool(float *feature_out, float *feature_in ,int featuremap_in_size,int poolling_size) {
	
	//定位寻址
	int out_layer_now = blockIdx.x;

	//定位到输出层的一层中的 x和y;
	int row_now = threadIdx.x;  // 表示所在层的行
	int col_now = threadIdx.y;  // 表示所在层的列

	//输出的feature尺寸
	int feature_out_size = featuremap_in_size / poolling_size;

	//定位到输出的地方
	int out_position = out_layer_now * feature_out_size*feature_out_size + row_now * feature_out_size + col_now;

	//定位到开始计算的featuremap地方
	int feature_map_start = out_layer_now * featuremap_in_size*featuremap_in_size + row_now*poolling_size * featuremap_in_size + col_now* poolling_size;

	//我们用 shared_Mem先装一下需要的数据
	
	/////////////////////////////////////////////////////////////////////////
	////   每个线程正式开始计算的算平面小区域内的数的  二维循环求最大   /////
	////////////////////////////////////////////////////////////////////////
	float big_value = 0.0;

	for (int i = 0; i < poolling_size; i++){
		for (int j = 0; j < poolling_size; j++)
		{	
		if (feature_in[feature_map_start + i * featuremap_in_size + j] > big_value) {
			big_value = feature_in[feature_map_start + i * featuremap_in_size + j];
		if (out_layer_now==2){
			printf(" ");
			}
			}
		} 
		__syncthreads();
	}
	__syncthreads();
	//把最大值赋值给  out的位置

	feature_out[out_position]= big_value;
}

//第三步做池化运算其实就是比大小
/*
float * feature_out      输出的feature_out指针
float * feature_in		 输入的feature_in指针  feature_out是feature_in的 1/n 倍，n的大小由池化核的大小决定
int featuremap_in_size   输入的featuremap的尺寸   输出的尺寸由输入的尺寸/倍数来得到
int poolling_size		 池化核的大小
*/
__global__ void pool_new(float *feature_out, float *feature_in, int featuremap_in_size, int poolling_size) {

	//定位寻址
	int out_layer_now = blockIdx.z;

	int threadraws_inablock = blockDim.x;
	int threadcols_inablock = blockDim.y;

	int blockraws_inagrid = gridDim.x;
	int blockcols_inagrid = gridDim.y;

	int block_now_raw = blockIdx.x;
	int block_now_col = blockIdx.y;

	//定位到输出层的一层中的 x和y;
	int row_now = (block_now_raw* threadraws_inablock)+threadIdx.x;  // 表示所在层的行
	int col_now = (block_now_col* threadcols_inablock)+threadIdx.y;  // 表示所在层的列

	//输出的feature尺寸
	int feature_out_size = featuremap_in_size / poolling_size;

	//定位到输出的地方
	int out_position = out_layer_now * feature_out_size*feature_out_size + row_now * feature_out_size + col_now;

	//定位到开始计算的featuremap地方
	int feature_map_start = out_layer_now * featuremap_in_size*featuremap_in_size + row_now * poolling_size * featuremap_in_size + col_now * poolling_size;

	//我们用 shared_Mem先装一下需要的数据

	/////////////////////////////////////////////////////////////////////////
	////   每个线程正式开始计算的算平面小区域内的数的  二维循环求最大   /////
	////////////////////////////////////////////////////////////////////////
	float big_value = 0.0f;
	__syncthreads();
	for (int i = 0; i < poolling_size; i++) {
		for (int j = 0; j < poolling_size; j++)
		{
			if (feature_in[feature_map_start + i * featuremap_in_size + j] >= big_value) {
				big_value = feature_in[feature_map_start + i * featuremap_in_size + j];
			}
			__syncthreads();
		}
		__syncthreads();
	}
	//把最大值赋值给  out的位置
	feature_out[out_position] = big_value;
}

/*
float *featuremap_in    输入的feature  输入的是一维
float *weight           权重得到形状为 M * N M为feature的大小 M为输入的向量的大小，N为输出的大小
float *feature_out      
int feature_in_size      输入的 数据的长度
相当于矩阵乘法
*/
__global__ void FC_SharedMem(float *featuremap_in, float *weight, float *feature_out,int feature_in_size) {

	int block_now= blockIdx.x;
	int thread_now = threadIdx.x;
	int threadsIn_A_Block = blockDim.x;

	//共享内存 每次取一个tile的数据
	__shared__ float FEATUREIN_tile[TILE_SIZE];
	__shared__ float WEIGHT_tile[TILE_SIZE][TILE_SIZE];

	//对于这个线程来 他取数据的初始位置  
	// 对应着	权重矩阵的第几行   因为我是一行表示一个与上一层连接的权重
	int start_pos = block_now * threadsIn_A_Block + thread_now;
	//  输出的结果的位置
	int out_pos = block_now* threadsIn_A_Block+ thread_now;

	// 计算要循环累加的次数
	int calculate_num = feature_in_size / TILE_SIZE;

	float result = 0.0f;

	//整个的过程就是装填 计算 在装填 再计算 最后把累加结果赋值
	//这就是一个线程所做的事
	for (int num = 0; num < calculate_num; num++)
	{
		/////////////////////////////////////////////////////////////////////
		////////      每一次tile计算的shared——mem						/////
		////////      因为是一个block共享shared_MEM所以千万不要赋值重了  ////
		/////////////////////////////////////////////////////////////////////
		//这一次相对于feature起点的偏移量
		int featurein_offset = num * TILE_SIZE;
		//对于feature_in来说每个线程只取一小块数据就好了
		//我们计算一下每个线程就近要搬多少个数据  TILE_SIZE 与 threadsIn_A_Block 最好成整倍数关系
		//比如tile_size=32 block人数为16  则每人办两次
		int dataperthread = TILE_SIZE / threadsIn_A_Block;

		//我们要做一些判定，判定每个线程一次从featrue中取多少凑成一个tile的数据
		for (int i = 0; i < dataperthread; i++)
		{
			FEATUREIN_tile[thread_now*dataperthread + i] = featuremap_in[num*TILE_SIZE + thread_now * dataperthread + i];
		}

		for (int i = 0; i < TILE_SIZE; i++)
		{
			//不出界判定
			if (featurein_offset + i < feature_in_size) {
				WEIGHT_tile[thread_now][i] = weight[start_pos*feature_in_size+ num * TILE_SIZE + i];
			}
		}
		__syncthreads();

		//计算一次tile的temp
		for (int i = 0; i < TILE_SIZE; i++)
		{
			result = result + FEATUREIN_tile[i] * WEIGHT_tile[thread_now][i];
		}
		__syncthreads();
	}
	printf("\n  result=%f 第%d个线程", result, out_pos);
	//最后再将球的数据传入out的数组中
	feature_out[out_pos] = result;
}


