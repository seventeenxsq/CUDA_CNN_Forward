#include "cuda_cnn.cuh"
#include"device_functions.h"

//  conv_step1ʵ��   ֻ�ǰ�ÿ��ľ����������Ӧ����
/*
float * step1_out   step1_out��featuremapָ��
float * feature_in   �����featuremapָ��
int con_core_in     ����˵�ָ��
int featuremap_size  featuremap�ߴ�  in��out�ĳߴ綼һ��
int coresize         ����˵ĳߴ�
*/
__global__ void conv_step1(float *step1_out, float * feature_in,
							float * con_core_in, int featuremap_size, int coresize)
{
	// �Ȳ��ܹ����ڴ�

	// ����ȡ���������ܸ���
	int layer_nums = gridDim.y;  // block�ĵڶ�άΪ�����Ҳ��feature�Ĳ���
	
	// ��λ���߳����ڵ�block
	int kernel_now = blockIdx.x;  //   Ŀǰ�ǵڼ���kernel
	int layer_now =  blockIdx.y;  //   Ŀǰ�ڵڼ�����

	// ��λ��block�е��߳�,Ҳ����kernel������ʵ��ִ����
	int row_now = threadIdx.x;  // ��ʾ���ڲ����
	int col_now =  threadIdx.y;  // ��ʾ���ڲ����


	//ÿ���߳�����������ʲô�أ�
	//���Ѷ�Ӧ����feature���ݺͶ�Ӧ�ľ�������
	// ��һ��3x3��Сѭ��

	//�������ڵ�Ҫ�����Ƕ�λ��featrue�еĶ�Ӧ������
	
	//�����out����Ķ�λ
	int out_position = (kernel_now*layer_nums + layer_now)*(featuremap_size*featuremap_size) + row_now * featuremap_size + col_now; //�������Ķ�λ
	
	// feature�е����ݶ�λ����λ���������ɣ����ÿ����ȡ
	int feature_layer_start= layer_now * (featuremap_size*featuremap_size);

	//  kernel��λ�������Ͳ���
	int kernel_pos_start = (kernel_now*layer_nums+layer_now)*coresize*coresize;

	//////////////////////////////////////////////////////////////
	////         ��ʼ����ѭ������һ�������ֵ                 ////
	//////////////////////////////////////////////////////////////
	float temp=0.0f;
	
	//��Ϊһ����������Ǵ������ϵı߽ǿ�ʼ��  ����Ҫ��λ�����ľ����ʼ���к���
	int starti = row_now - coresize / 2;
	int startj = col_now - coresize / 2;

	for (int i = starti; i < starti + coresize; i++){
		for (int j = startj; j < startj + coresize; j++){
			if (i >= 0 && j >= 0 && i < featuremap_size && j < featuremap_size)
			{
				temp =temp+con_core_in[kernel_pos_start+(i - starti)*coresize+(j - startj)] * feature_in[feature_layer_start+(i * featuremap_size)+j];
				//printf(" con_core_in[%d %d] = %f feature_in[%d %d] =%f\n",i,j, con_core_in[kernel_pos_start + (i - starti)*coresize + (j - startj)],i,j, feature_in[feature_layer_start + (i * featuremap_size) + j]);
			}
		}

	}
	printf("\n temp= %f \n", temp);
	step1_out[out_position] = temp;
}

//�ڶ��������ۼӣ��������޹��� ֻ��������ۼ�
/*
float * feature_out   �����featureָ��
float * step1_out    conv_step1�Ľ������
int core_num     ����˸���   ��Ӧ�����featuremap�Ĳ���
int core_layers  ����˲���   ��Ӧ��ÿ���ۼ���Ҫ�����ٸ�����������ܵ��ۼӡ�
*/
__global__ void conv_step2(float * feature_out, float * step1_out,
				int core_num, int core_layers,int step1_out_featuremap_size) {
	// ��λ�� ������ ���
	int out_layer_now = blockIdx.x;

	//��λ��������һ���е� x��y ;
	int row_now = threadIdx.x;  // ��ʾ���ڲ����
	int col_now = threadIdx.y;  // ��ʾ���ڲ����

	//�ҵ�ÿ���ۼӼ��㿪ʼ�ĵط�
	int start_pos=out_layer_now * core_layers * (step1_out_featuremap_size*step1_out_featuremap_size)
						+ row_now*step1_out_featuremap_size+col_now;
	
	//��������λ��
	int out_pos=out_layer_now*step1_out_featuremap_size*step1_out_featuremap_size+ row_now*step1_out_featuremap_size+col_now;

	float temp=0.0f;

	/////////////////// ��ʽѭ�����㲿��  ///////////////////
	//��ѭ����ʼ�ۼ����� ������һ�����ֵ
	for(int i=0;i<core_layers;i++){
		temp= temp + step1_out[start_pos+i*step1_out_featuremap_size*step1_out_featuremap_size];
	}
	///////////////////////////////////////////////////////

	feature_out[out_pos] = temp;
}

//��һ���͵ڶ�����������һ��������	�������

//���������ػ�������ʵ���Ǳȴ�С
/*
float * feature_out      �����feature_outָ��
float * feature_in		 �����feature_inָ��  feature_out��feature_in�� 1/n ����n�Ĵ�С�ɳػ��˵Ĵ�С����
int featuremap_in_size   �����featuremap�ĳߴ�   ����ĳߴ�������ĳߴ�/�������õ�  
int poolling_size		 �ػ��˵Ĵ�С
*/
__global__ void pool(float *feature_out, float *feature_in ,int featuremap_in_size,int poolling_size) {
	
	//��λѰַ
	int out_layer_now = blockIdx.x;

	//��λ��������һ���е� x��y;
	int row_now = threadIdx.x;  // ��ʾ���ڲ����
	int col_now = threadIdx.y;  // ��ʾ���ڲ����

	//�����feature�ߴ�
	int feature_out_size = featuremap_in_size / poolling_size;

	//��λ������ĵط�
	int out_position = out_layer_now * feature_out_size*feature_out_size + row_now * feature_out_size + col_now;

	//��λ����ʼ�����featuremap�ط�
	int feature_map_start = out_layer_now * featuremap_in_size*featuremap_in_size + row_now*poolling_size * featuremap_in_size + col_now* poolling_size;

	//������ shared_Mem��װһ����Ҫ������
	
	/////////////////////////////////////////////////////////////////////////
	////   ÿ���߳���ʽ��ʼ�������ƽ��С�����ڵ�����  ��άѭ�������   /////
	////////////////////////////////////////////////////////////////////////
	float big_value = 0.0f;

	for (int i = 0; i < poolling_size; i++){
		for (int j = 0; j < poolling_size; j++)
		{	
		if (feature_in[feature_map_start + i * featuremap_in_size + j] >= big_value) {
			big_value = feature_in[feature_map_start + i * featuremap_in_size + j];
		}
		__syncthreads();
		} 
	}
	//�����ֵ��ֵ��  out��λ��
	feature_out[out_position]= big_value;
}

/*
float *featuremap_in    �����feature  �������һά
float *weight           Ȩ�صõ���״Ϊ M * N MΪfeature�Ĵ�С MΪ����������Ĵ�С��NΪ����Ĵ�С
float *feature_out      
int feature_in_size      ����� ���ݵĳ���
�൱�ھ���˷�
*/
__global__ void FC_SharedMem(float *featuremap_in, float *weight, float *feature_out,int feature_in_size) {

	int block_now= blockIdx.x;
	int thread_now = threadIdx.x;
	int threadsIn_A_Block = blockDim.x;

	//�����ڴ� ÿ��ȡһ��tile������
	__shared__ float FEATUREIN_tile[TILE_SIZE];
	__shared__ float WEIGHT_tile[TILE_SIZE][TILE_SIZE];

	//��������߳��� ��ȡ���ݵĳ�ʼλ��  
	// ��Ӧ��	Ȩ�ؾ���ĵڼ���   ��Ϊ����һ�б�ʾһ������һ�����ӵ�Ȩ��
	int start_pos = block_now * threadsIn_A_Block + thread_now;
	//  ����Ľ����λ��
	int out_pos = block_now* threadsIn_A_Block+ thread_now;

	// ����Ҫѭ���ۼӵĴ���
	int calculate_num = feature_in_size / TILE_SIZE;

	float result = 0.0f;

	//�����Ĺ��̾���װ�� ���� ��װ�� �ټ��� �����ۼӽ����ֵ
	//�����һ���߳���������
	for (int num = 0; num < calculate_num; num++)
	{
		/////////////////////////////////////////////////////////////////////
		////////      ÿһ��tile�����shared����mem						/////
		////////      ��Ϊ��һ��block����shared_MEM����ǧ��Ҫ��ֵ����  ////
		/////////////////////////////////////////////////////////////////////
		//��һ�������feature����ƫ����
		int featurein_offset = num * TILE_SIZE;
		//����feature_in��˵ÿ���߳�ֻȡһС�����ݾͺ���
		//���Ǽ���һ��ÿ���߳̾ͽ�Ҫ����ٸ�����  TILE_SIZE �� threadsIn_A_Block ��ó���������ϵ
		//����tile_size=32 block����Ϊ16  ��ÿ�˰�����
		int dataperthread = TILE_SIZE / threadsIn_A_Block;

		//����Ҫ��һЩ�ж����ж�ÿ���߳�һ�δ�featrue��ȡ���ٴճ�һ��tile������
		for (int i = 0; i < dataperthread; i++)
		{
			FEATUREIN_tile[thread_now*dataperthread + i] = featuremap_in[num*TILE_SIZE + thread_now * dataperthread + i];
		}

		for (int i = 0; i < TILE_SIZE; i++)
		{
			//�������ж�
			if (featurein_offset + i < feature_in_size) {
				WEIGHT_tile[thread_now][i] = weight[start_pos*feature_in_size+ num * TILE_SIZE + i];
			}
		}
		__syncthreads();

		//����һ��tile��temp
		for (int i = 0; i < TILE_SIZE; i++)
		{
			result = result + FEATUREIN_tile[i] * WEIGHT_tile[thread_now][i];
		}
		__syncthreads();
	}
	//����ٽ�������ݴ���out��������
	feature_out[out_pos] = result;
}
