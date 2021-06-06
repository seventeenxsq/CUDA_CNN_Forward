#include "cuda_cnn.cuh"

#include <opencv2/opencv.hpp>
#include <opencv2\imgproc\types_c.h>
using namespace cv;
using namespace std;

//����������һ����ά�ľ������
/*
  �����������һ�� 16*16�ľ���
  ����˴�С Ϊ3
  �����Ҳ�� 16*16�ľ���
  threadsperblock(16,16)

  ��Ӧ�ô���ȥ�Ĳ�����
  ����ĵ�ַ
  �������ĵ�ַ
  ����˵ĵ�ַ
  ����˵Ŀ��
  ����ĳ��Ϳ�
 */

__global__ void conv(float *in, float *out, float *mask, int maskwidth, int w, int h) {
	// ��Ҫȷ�������� ֪��������߳����ڸ�ʲô
	int raw = threadIdx.x;
	int col = threadIdx.y;

	int startx = raw - maskwidth / 2;
	int starty = col - maskwidth / 2;

	float tempvalue;

	//����ڲ�����
	for (int i = startx; i < startx + maskwidth; i++)
	{
		for (int j = starty; j < starty + maskwidth; j++)
		{
			//����tempvalueʱ��Ҫ�ж�����Խ��
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
	//ȷ����Ҫ���ɵ�feature��С
	int arr_size_3D =3* 5*5;
	//kernel�Ĵ�С
	int kernel_3d = 10 * 3 * 3 * 3; 
	//step1_out��С
	int step1_out_size = 10 * 3 * 5 * 5;

	/*
	��������ָ��ʱ ������Ҫ��������
	һ����CPU�е�����ָ��  һ���Ǵ���GPU�е�����ָ��
	Ϊ������������ֱ�Ӷ�����һά����
	*/

	// ������cpu�е� �������鲢��ֵ
	float *feature_in_cpu = (float*)malloc(arr_size_3D * sizeof(float));
	float *kernel_in_cpu = (float*)malloc(kernel_3d * sizeof(float));
	float *feature_out_cpu = (float*)malloc(step1_out_size * sizeof(float));

	////////////////////////////////////////////////////////////////////
	//-------------------  ���ݸ�ֵ���� -----------------------
	//feature��ֵ
	for(int layer=0;layer<3;layer++){
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				feature_in_cpu[layer*5*5+i*5+j] = layer+1;
			}
		}
	}

	//kernel ��ֵ  4άѭ��
	for(int kernel_num=0;kernel_num<10;kernel_num++){
		for(int layer=0;layer<3;layer++){
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					kernel_in_cpu[(kernel_num*3+layer)*3*3+i*3+j] = kernel_num+1;
				}
			}
		}
	}	

	//-----------------------------���ݸ�ֵ���------------------------------
	//////////////////////////////////////////////////////////////////////////

	////���featrue��� ������
	printf(" \n featrue��ʼ�� \n ");
	for(int layer=0;layer<3;layer++){
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				printf("%f\t", feature_in_cpu[layer*5*5+i*5+j]);
			}
		}
		printf("\n");
	}

	printf("--------------------------------------------------------");

	////���kernel��� ������  4ά
	printf(" \n kernel�ĳ�ʼ�� \n ");
	for(int kernel_num=0;kernel_num<10;kernel_num++){
		for(int layer=0;layer<3;layer++){
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					printf("%f\t", kernel_in_cpu[(kernel_num*3+layer)*3*3+i*3+j]);
				}
			}
			printf("\n\n");
		}
		printf("------------- �� %d ����������� --------------\n", kernel_num);
	}

	printf("\n ��ʼ��һ������������\n");

	//����GPU������ָ�� define GPU value pointer 
	float *feature_in_gpu, *kernel_in_gpu, *out_gpu;
	//���Ƕ���Ҫ����GPUcache�е�����

	// �� GPU�� ��cudaMalloc�����ڴ�ռ�
	//һ��Ҫ�ǵô���ȥ ����ָ��
	cudaMalloc((void**)&feature_in_gpu, sizeof(float)*arr_size_3D);
	cudaMalloc((void**)&kernel_in_gpu, sizeof(float)*kernel_3d);
	cudaMalloc((void**)&out_gpu, sizeof(float)*step1_out_size);

	// ��CPU�е����ݴ��� GPU�ڴ���
	cudaMemcpy(feature_in_gpu, feature_in_cpu, sizeof(float)*arr_size_3D, cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_in_gpu, kernel_in_cpu, sizeof(float)*kernel_3d, cudaMemcpyHostToDevice);

	printf("\n �ڴ�COPY��� \n");
	//������ ������

	//��ƴ�����߳�  �����߳�
	dim3 dimBlock(5, 5);// �̵߳���״
	dim3 dimGrid(10, 3);

	//����CUDA APIʱ����Ӧ�ô������device���ڴ�ָ�룡����
	//����ֱ�Ӿ͵��ò���kernel
	conv_step1<<<dimGrid,dimBlock>>>(out_gpu, feature_in_gpu, kernel_in_gpu, 5, 3);
	//��GPU����õ��ڴ淵�ظ�CPU
	//copy back
	cudaMemcpy(feature_out_cpu, out_gpu, sizeof(float)*step1_out_size, cudaMemcpyDeviceToHost);

	////���featrue_out_cpu ��� ������
	printf("---------------------------------------\n");
	printf("-------���featrue_out_cpu���-----------\n");
	printf("---------------------------------------\n");
	for(int kernel_num=0; kernel_num <10; kernel_num++){
		for (int layer = 0; layer < 3; layer++){
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 5; j++) {
					printf("%f\t", feature_out_cpu[(kernel_num*3+layer) * 5 * 5 + i * 5 + j]);
				}
			}
			printf(" \n             һ�����ֵ        \n");
		}
		printf("\n---------------------------------\n");
	}

	printf("\n  --------- conv step 1 ȫ����� ------ \n");

	cudaFree(feature_in_gpu);
	cudaFree(kernel_in_gpu);
	cudaFree(out_gpu);
	printf("\n  --------- �ͷ�ָ����� ------ \n");

}

void test_conv_step1_new() {
	//ȷ����Ҫ���ɵ�feature��С
	int arr_size_3D = 3 * 5 * 5;
	//kernel�Ĵ�С
	int kernel_3d = 10 * 3 * 3 * 3;
	//step1_out��С
	int step1_out_size = 10 * 3 * 5 * 5;

	/*
	��������ָ��ʱ ������Ҫ��������
	һ����CPU�е�����ָ��  һ���Ǵ���GPU�е�����ָ��
	Ϊ������������ֱ�Ӷ�����һά����
	*/

	// ������cpu�е� �������鲢��ֵ
	float *feature_in_cpu = (float*)malloc(arr_size_3D * sizeof(float));
	float *kernel_in_cpu = (float*)malloc(kernel_3d * sizeof(float));
	float *feature_out_cpu = (float*)malloc(step1_out_size * sizeof(float));

	////////////////////////////////////////////////////////////////////
	//-------------------  ���ݸ�ֵ���� -----------------------
	//feature��ֵ
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				feature_in_cpu[layer * 5 * 5 + i * 5 + j] = layer + 1;
			}
		}
	}

	//kernel ��ֵ  4άѭ��
	for (int kernel_num = 0; kernel_num < 10; kernel_num++) {
		for (int layer = 0; layer < 3; layer++) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					kernel_in_cpu[(kernel_num * 3 + layer) * 3 * 3 + i * 3 + j] = kernel_num + 1;
				}
			}
		}
	}

	//-----------------------------���ݸ�ֵ���------------------------------
	//////////////////////////////////////////////////////////////////////////

	////���featrue��� ������
	printf(" \n featrue��ʼ�� \n ");
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				printf("%f\t", feature_in_cpu[layer * 5 * 5 + i * 5 + j]);
			}
		}
		printf("\n");
	}

	printf("--------------------------------------------------------");

	////���kernel��� ������  4ά
	printf(" \n kernel�ĳ�ʼ�� \n ");
	for (int kernel_num = 0; kernel_num < 10; kernel_num++) {
		for (int layer = 0; layer < 3; layer++) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					printf("%f\t", kernel_in_cpu[(kernel_num * 3 + layer) * 3 * 3 + i * 3 + j]);
				}
			}
			printf("\n\n");
		}
		printf("------------- �� %d ����������� --------------\n", kernel_num);
	}

	printf("\n ��ʼ��һ������������\n");

	//����GPU������ָ�� define GPU value pointer 
	float *feature_in_gpu, *kernel_in_gpu, *out_gpu;
	//���Ƕ���Ҫ����GPUcache�е�����

	// �� GPU�� ��cudaMalloc�����ڴ�ռ�
	//һ��Ҫ�ǵô���ȥ ����ָ��
	cudaMalloc((void**)&feature_in_gpu, sizeof(float)*arr_size_3D);
	cudaMalloc((void**)&kernel_in_gpu, sizeof(float)*kernel_3d);
	cudaMalloc((void**)&out_gpu, sizeof(float)*step1_out_size);

	// ��CPU�е����ݴ��� GPU�ڴ���
	cudaMemcpy(feature_in_gpu, feature_in_cpu, sizeof(float)*arr_size_3D, cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_in_gpu, kernel_in_cpu, sizeof(float)*kernel_3d, cudaMemcpyHostToDevice);

	printf("\n �ڴ�COPY��� \n");
	//������ ������

	//��ƴ�����߳�  �����߳�
	dim3 dimGrid(1,1,10);
	dim3 dimBlock(5,5);// �̵߳���״
	
	//����CUDA APIʱ����Ӧ�ô������device���ڴ�ָ�룡����
	//����ֱ�Ӿ͵��ò���kernel
	conv_step1_new<<<dimGrid, dimBlock >>>(out_gpu, feature_in_gpu, kernel_in_gpu, 5, 3,3);
	//��GPU����õ��ڴ淵�ظ�CPU
	//copy back
	cudaMemcpy(feature_out_cpu, out_gpu, sizeof(float)*step1_out_size, cudaMemcpyDeviceToHost);

	////���featrue_out_cpu ��� ������
	printf("---------------------------------------\n");
	printf("-------���featrue_out_cpu���-----------\n");
	printf("---------------------------------------\n");
	for (int kernel_num = 0; kernel_num < 10; kernel_num++) {
		for (int layer = 0; layer < 3; layer++) {
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 5; j++) {
					printf("%f\t", feature_out_cpu[(kernel_num * 3 + layer) * 5 * 5 + i * 5 + j]);
				}
			}
			printf(" \n             һ�����ֵ        \n");
		}
		printf("\n---------------------------------\n");
	}

	printf("\n  --------- conv step 1 ȫ����� ------ \n");

	cudaFree(feature_in_gpu);
	cudaFree(kernel_in_gpu);
	cudaFree(out_gpu);
	printf("\n  --------- �ͷ�ָ����� ------ \n");

}

void test_conv_step2() {
	//ȷ����Ҫ���ɵ�feature��С
	int arr_size_3D = 9 * 5 * 5;   //3 ������� *3��=9  

	//step2_out��С
	int step2_out_size =  3 * 5 * 5;

	/*
	��������ָ��ʱ ������Ҫ��������
	һ����CPU�е�����ָ��  һ���Ǵ���GPU�е�����ָ��
	Ϊ��������ֱ�Ӷ�����һά����
	*/

	// ������cpu�е� �������鲢��ֵ
	float *feature_in_cpu = (float*)malloc(arr_size_3D * sizeof(float));
	float *feature_out_cpu = (float*)malloc(step2_out_size * sizeof(float));

	////////////////////////////////////////////////////////////////////
	//-------------------  ���ݸ�ֵ���� -----------------------
	//feature��ֵ
	for (int kernel = 0; kernel < 3; kernel++){
		for (int layer = 0; layer < 3; layer++){
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 5; j++) {
					feature_in_cpu[kernel* 3 * 5 * 5+ layer*5*5 + i * 5 + j] = kernel + 1;
				}
			}
		}
	}


	//kernel ��ֵ  4άѭ��
	/*for (int kernel_num = 0; kernel_num < 10; kernel_num++) {
		for (int layer = 0; layer < 3; layer++) {
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					kernel_in_cpu[(kernel_num * 3 + layer) * 3 * 3 + i * 3 + j] = kernel_num + 1;
				}
			}
		}
	}*/

	//-----------------------------���ݸ�ֵ���------------------------------
	//////////////////////////////////////////////////////////////////////////


	////���featrue��� ������
	printf(" \n featrue_step1 ��ʼ�� \n ");
	for (int kernel = 0; kernel < 3; kernel++) {
		for (int layer = 0; layer < 3; layer++) {
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 5; j++) {
					printf("%f\t", feature_in_cpu[kernel * 3 * 5 * 5 + layer * 5 * 5 + i * 5 + j]);
				}
			}
			printf("\n");
		}
		printf("------------- �� %d ����������� --------------\n", kernel);
	}
	printf("\n------------------------feature_out ��ʼ�����--------------------------------\n");
	printf("\n ��ʼ��һ������������\n");
	//-------------------------------------------------------------------------------------------------

	//����GPU������ָ�� define GPU value pointer
	float *feature_in_gpu, *out_gpu;
	//���Ƕ���Ҫ����GPUcache�е�����

	// �� GPU�� ��cudaMalloc�����ڴ�ռ�
	//һ��Ҫ�ǵô���ȥ ����ָ��
	cudaMalloc((void**)&feature_in_gpu, sizeof(float)*arr_size_3D);
	cudaMalloc((void**)&out_gpu, sizeof(float)*step2_out_size);

	// ��CPU�е����ݴ��� GPU�ڴ���
	cudaMemcpy(feature_in_gpu, feature_in_cpu, sizeof(float)*arr_size_3D, cudaMemcpyHostToDevice);

	printf("\n �ڴ�COPY��� \n");

	////////////////////////////////////////////////////////////////////////////////////////////
	/////////      ǰ������׼����  ����������߳̽ṹ����kernel����                     ////////
	///////////////////////////////////////////////////////////////////////////////////////////
	//��ƴ�����߳�  �����߳�
	dim3 dimBlock(5, 5);// �̵߳���״
	dim3 dimGrid(3);

	//����CUDA APIʱ����Ӧ�ô������device���ڴ�ָ�룡����
	//����ֱ�Ӿ͵��ò���kernel
	conv_step2<<<dimGrid, dimBlock>>>(out_gpu, feature_in_gpu, 3,3,5);
	
	//��GPU����õ��ڴ淵�ظ�CPU
	//copy back from gpu
	cudaMemcpy(feature_out_cpu, out_gpu, sizeof(float)*step2_out_size, cudaMemcpyDeviceToHost);

	////// ���featrue_out_cpu ��� ������   //////////
	printf("---------------------------------------\n");
	printf("-------���featrue_out_cpu���-----------\n");
	printf("---------------------------------------\n");
	//for (int kernel_num = 0; kernel_num < 10; kernel_num++) {
		for (int layer = 0; layer < 3; layer++) {
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 5; j++) {
					printf("%f\t", feature_out_cpu[layer * 5 * 5 + i * 5 + j]);
				}
			}
			printf(" \n             һ�����ֵ        \n");
		}
		printf("\n---------------------------------\n");
	//}

	printf("\n  --------- conv step 2 ȫ����� ------ \n");

	cudaFree(feature_in_gpu);
	cudaFree(out_gpu);
	printf("\n  --------- �ͷ�ָ����� ------ \n");

}

void test_conv_step2_new() {
	//ȷ����Ҫ���ɵ�feature��С
	int arr_size_3D = 3* 3 * 64 * 64;   //3 ������� *3��=9  

	//step2_out��С
	int step2_out_size = 3 * 64 * 64;

	/*
	��������ָ��ʱ ������Ҫ��������
	һ����CPU�е�����ָ��  һ���Ǵ���GPU�е�����ָ��
	Ϊ��������ֱ�Ӷ�����һά����
	*/

	// ������cpu�е� �������鲢��ֵ
	float *feature_in_cpu = (float*)malloc(arr_size_3D * sizeof(float));
	float *feature_out_cpu = (float*)malloc(step2_out_size * sizeof(float));

	////////////////////////////////////////////////////////////////////
	//-------------------  ���ݸ�ֵ���� -----------------------
	//feature��ֵ
	for (int kernel = 0; kernel < 3; kernel++) {
		for (int layer = 0; layer < 3; layer++) {
			for (int i = 0; i < 64; i++) {
				for (int j = 0; j < 64; j++) {
					feature_in_cpu[kernel * 3 * 64 * 64 + layer * 64 * 64 + i * 64 + j] = kernel + 1;
				}
			}
		}
	}

	//-----------------------------���ݸ�ֵ���------------------------------
	//////////////////////////////////////////////////////////////////////////


	////���featrue��� ������
	printf(" \n featrue_step1 ��ʼ�� \n ");
	for (int kernel = 0; kernel < 3; kernel++) {
		for (int layer = 0; layer < 3; layer++) {
			for (int i = 0; i < 64; i++) {
				for (int j = 0; j < 64; j++) {
					printf("%f\t", feature_in_cpu[kernel * 3 * 64 * 64 + layer * 64 * 64 + i * 64 + j]);
				}
			}
			printf("\n");
		}
		printf("------------- �� %d ����������� --------------\n", kernel);
	}
	printf("\n------------------------feature_out ��ʼ�����--------------------------------\n");
	printf("\n ��ʼ��һ������������\n");
	//-------------------------------------------------------------------------------------------------

	//����GPU������ָ�� define GPU value pointer
	float *feature_in_gpu, *out_gpu;
	//���Ƕ���Ҫ����GPUcache�е�����

	// �� GPU�� ��cudaMalloc�����ڴ�ռ�
	//һ��Ҫ�ǵô���ȥ ����ָ��
	cudaMalloc((void**)&feature_in_gpu, sizeof(float)*arr_size_3D);
	cudaMalloc((void**)&out_gpu, sizeof(float)*step2_out_size);

	// ��CPU�е����ݴ��� GPU�ڴ���
	cudaMemcpy(feature_in_gpu, feature_in_cpu, sizeof(float)*arr_size_3D, cudaMemcpyHostToDevice);

	printf("\n �ڴ�COPY��� \n");

	////////////////////////////////////////////////////////////////////////////////////////////
	/////////      ǰ������׼����  ����������߳̽ṹ����kernel����                     ////////
	///////////////////////////////////////////////////////////////////////////////////////////
	//��ƴ�����߳�  �����߳�
	dim3 dimGrid(2,2,3);
	dim3 dimBlock(32, 32);// �̵߳���״

	//����CUDA APIʱ����Ӧ�ô������device���ڴ�ָ�룡����
	//����ֱ�Ӿ͵��ò���kernel
	conv_step2_new<<<dimGrid, dimBlock >> > (out_gpu, feature_in_gpu, 3, 64);

	//��GPU����õ��ڴ淵�ظ�CPU
	//copy back from gpu
	cudaMemcpy(feature_out_cpu, out_gpu, sizeof(float)*step2_out_size, cudaMemcpyDeviceToHost);

	////// ���featrue_out_cpu ��� ������   //////////
	printf("---------------------------------------\n");
	printf("-------���featrue_out_cpu���-----------\n");
	printf("---------------------------------------\n");
	//for (int kernel_num = 0; kernel_num < 10; kernel_num++) {
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 64; i++) {
			for (int j = 0; j < 64; j++) {
				printf("%f\t", feature_out_cpu[layer * 64 * 64 + i * 64 + j]);
			}
		}
		printf(" \n             һ�����ֵ        \n");
	}
	printf("\n---------------------------------\n");
	//}

	printf("\n  --------- conv step 2 ȫ����� ------ \n");

	cudaFree(feature_in_gpu);
	cudaFree(out_gpu);
	printf("\n  --------- �ͷ�ָ����� ------ \n");
}

void test_pool() {
	//ȷ����Ҫ���ɵ�feature��С
	int arr_size_3D = 3 * 8 * 8;   //3�� ÿ���feature_mapΪ 8*8��С  

	//step2_out��С
	int pool_out_size = 3 * 4 * 4;

	/*
	��������ָ��ʱ ������Ҫ��������
	һ����CPU�е�����ָ��  һ���Ǵ���GPU�е�����ָ��
	Ϊ��������ֱ�Ӷ�����һά����
	*/

	// ������cpu�е� �������鲢��ֵ
	float *feature_in_cpu = (float*)malloc(arr_size_3D * sizeof(float));
	float *pool_out_cpu = (float*)malloc(pool_out_size * sizeof(float));

	////////////////////////////////////////////////////////////////////
	//-------------------  ���ݸ�ֵ���� -----------------------
	//feature��ֵ
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++){
				feature_in_cpu[layer * 8 * 8 + i *8 + j] = 1.0*(i+j);
			}
		}
	}

	//-----------------------------���ݸ�ֵ���------------------------------
	//////////////////////////////////////////////////////////////////////////

	////���featrue��� ������
	printf(" \n featrue_in ��ʼ�� \n ");
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				printf("%f\t",feature_in_cpu[layer * 8 * 8 + i * 8 + j]);
				}
			printf("\n");
		}
		printf("------------- �� %d �� --------------\n", layer);
	}
	printf("\n------------------------feature_in ��ʼ�����--------------------------------\n");
	printf("\n ��ʼ��һ������������\n");
	//-------------------------------------------------------------------------------------------------

	//����GPU������ָ�� define GPU value pointer
	float *feature_in_gpu, *out_gpu;
	//���Ƕ���Ҫ����GPUcache�е�����

	//// �� GPU�� ��cudaMalloc�����ڴ�ռ�
	////һ��Ҫ�ǵô���ȥ ����ָ��
	cudaMalloc((void**)&feature_in_gpu, sizeof(float)*arr_size_3D);
	cudaMalloc((void**)&out_gpu, sizeof(float)*pool_out_size);

	// ��CPU�е����ݴ��� GPU�ڴ���
	cudaMemcpy(feature_in_gpu, feature_in_cpu, sizeof(float)*arr_size_3D, cudaMemcpyHostToDevice);
	cudaError_t result = cudaMemcpy(feature_in_gpu, feature_in_cpu, sizeof(float)*arr_size_3D, cudaMemcpyHostToDevice);

	printf("\n #########�ڴ�COPY���  %d ########## \n",result);

	////////////////////////////////////////////////////////////////////////////////////////////
	/////////      ǰ������׼����  ����������߳̽ṹ����kernel����                     ////////
	///////////////////////////////////////////////////////////////////////////////////////////
	//��ƴ�����߳�  �����߳�
	dim3 dimBlock(4, 4);// �̵߳���״
	dim3 dimGrid(3);

	////����CUDA APIʱ����Ӧ�ô������device���ڴ�ָ�룡����
	////����ֱ�Ӿ͵��ò���kernel
	pool<<<dimGrid, dimBlock>>> (out_gpu, feature_in_gpu, 8, 2);

	//��GPU����õ��ڴ淵�ظ�CPU
	//copy back from gpu
	cudaMemcpy(pool_out_cpu, out_gpu, sizeof(float)*pool_out_size, cudaMemcpyDeviceToHost);

	//////// ���featrue_out_cpu ��� ������   //////////
	printf("---------------------------------------\n");
	printf("-------���featrue_out_cpu���-----------\n");
	printf("---------------------------------------\n");
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				printf("%f\t", pool_out_cpu[layer * 4 * 4 + i * 4 + j]);
			}
		}
		printf(" \n             һ�����ֵ        \n");
	}
	printf("\n---------------------------------\n");
	//}

	cudaFree(feature_in_gpu);
	cudaFree(out_gpu);
	printf("\n  --------- �ͷ�ָ����� ------ \n");

}

void test_pool_new() {
	//ȷ����Ҫ���ɵ�feature��С
	int arr_size_3D = 3 * 8 * 8;   //3�� ÿ���feature_mapΪ 8*8��С  

	//step2_out��С
	int pool_out_size = 3 * 4 * 4;

	/*
	��������ָ��ʱ ������Ҫ��������
	һ����CPU�е�����ָ��  һ���Ǵ���GPU�е�����ָ��
	Ϊ��������ֱ�Ӷ�����һά����
	*/

	// ������cpu�е� �������鲢��ֵ
	float *feature_in_cpu = (float*)malloc(arr_size_3D * sizeof(float));
	float *pool_out_cpu = (float*)malloc(pool_out_size * sizeof(float));

	////////////////////////////////////////////////////////////////////
	//-------------------  ���ݸ�ֵ���� -----------------------
	//feature��ֵ
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				feature_in_cpu[layer * 8 * 8 + i * 8 + j] = i + j;
			}
		}
	}

	//-----------------------------���ݸ�ֵ���------------------------------
	//////////////////////////////////////////////////////////////////////////

	////���featrue��� ������
	printf(" \n featrue_in ��ʼ�� \n ");
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 8; j++) {
				printf("%f\t", feature_in_cpu[layer * 8 * 8 + i * 8 + j]);
			}
			printf("\n");
		}
		printf("------------- �� %d �� --------------\n", layer);
	}
	printf("\n------------------------feature_in ��ʼ�����--------------------------------\n");
	printf("\n ��ʼ��һ������������\n");
	//-------------------------------------------------------------------------------------------------

	//����GPU������ָ�� define GPU value pointer
	float *feature_in_gpu, *out_gpu;
	//���Ƕ���Ҫ����GPUcache�е�����

	//// �� GPU�� ��cudaMalloc�����ڴ�ռ�
	////һ��Ҫ�ǵô���ȥ ����ָ��
	cudaMalloc((void**)&feature_in_gpu, sizeof(float)*arr_size_3D);
	cudaMalloc((void**)&out_gpu, sizeof(float)*pool_out_size);

	// ��CPU�е����ݴ��� GPU�ڴ���
	cudaMemcpy(feature_in_gpu, feature_in_cpu, sizeof(float)*arr_size_3D, cudaMemcpyHostToDevice);

	//printf("\n �ڴ�COPY��� \n");

	////////////////////////////////////////////////////////////////////////////////////////////
	/////////      ǰ������׼����  ����������߳̽ṹ����kernel����                     ////////
	///////////////////////////////////////////////////////////////////////////////////////////
	//��ƴ�����߳�  �����߳�
	dim3 dimBlock(4, 4);// �̵߳���״
	dim3 dimGrid(1,1,3);

	////����CUDA APIʱ����Ӧ�ô������device���ڴ�ָ�룡����
	////����ֱ�Ӿ͵��ò���kernel
	pool_new<<<dimGrid, dimBlock>>> (out_gpu, feature_in_gpu, 8, 2);

	//��GPU����õ��ڴ淵�ظ�CPU
	//copy back from gpu
	cudaMemcpy(pool_out_cpu, out_gpu, sizeof(float)*pool_out_size, cudaMemcpyDeviceToHost);

	//////// ���featrue_out_cpu ��� ������   //////////
	printf("---------------------------------------\n");
	printf("-------���featrue_out_cpu���-----------\n");
	printf("---------------------------------------\n");
	for (int layer = 0; layer < 3; layer++) {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				printf("%f\t", pool_out_cpu[layer * 4 * 4 + i * 4 + j]);
			}
		}
		printf(" \n             һ�����ֵ        \n");
	}
	printf("\n---------------------------------\n");
	//}

	cudaFree(feature_in_gpu);
	cudaFree(out_gpu);
	printf("\n  --------- �ͷ�ָ����� ------ \n");

}

void test_FC_SharedMem() {
	//ȷ����Ҫ���ɵ�feature��С
	int arr_size_3D = 128 ;
	//kernel�Ĵ�С
	int weight_size = 128*256 ; 
	//step1_out��С
	int featureout_size = 256 ;

	float *feature_in_cpu = (float*)malloc(arr_size_3D * sizeof(float));
	float *weight_in_cpu = (float*)malloc(weight_size * sizeof(float));
	float *feature_out_cpu = (float*)malloc(featureout_size * sizeof(float));

	////////////////////////////////////////////////////////////////////
	//-------------------  ���ݸ�ֵ���� -----------------------
	//feature��ֵ
	for(int i=0; i < arr_size_3D ;i++){
		feature_in_cpu[i] = i+1;
	}

	//kernel ��ֵ  
	for(int raw=0;raw<256;raw++){
		for(int col=0;col<128;col++){
			weight_in_cpu[raw*128+col] = raw;
		}
	}

	//������Խ��
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

	//����GPU������ָ�� define GPU value pointer 
	float *feature_in_gpu, *weight_in_gpu, *out_gpu;
	//���Ƕ���Ҫ����GPU globalMEm�е�����

	// �� GPU�� ��cudaMalloc�����ڴ�ռ�
	//һ��Ҫ�ǵô���ȥ ����ָ��
	cudaMalloc((void**)&feature_in_gpu, sizeof(float)*arr_size_3D);
	cudaMalloc((void**)&weight_in_gpu, sizeof(float)*weight_size);
	cudaMalloc((void**)&out_gpu, sizeof(float)*featureout_size);

	// ��CPU�е����ݴ��� GPU�ڴ���
	cudaMemcpy(feature_in_gpu, feature_in_cpu, sizeof(float)*arr_size_3D, cudaMemcpyHostToDevice);
	cudaMemcpy(weight_in_gpu, weight_in_cpu, sizeof(float)*weight_size, cudaMemcpyHostToDevice);

	//��ƴ�����߳�  �����߳�
	dim3 dimGrid(16);// �̵߳���״
	dim3 dimBlock(16);

	FC_SharedMem<<<dimGrid, dimBlock>>>(feature_in_gpu, weight_in_gpu, out_gpu, 128);

	cudaMemcpy(feature_out_cpu, out_gpu, sizeof(float)*featureout_size, cudaMemcpyDeviceToHost);

	////���featrue_out_cpu ��� ������
	printf("---------------------------------------\n");
	printf("-------���featrue_out_cpu���-----------\n");
	printf("---------------------------------------\n");
	for(int i=0; i <featureout_size; i++){
		if (i % 32 == 0)  printf("\n");
		printf("%f\t", feature_out_cpu[i]);
	}
	cudaFree(feature_in_gpu);
	cudaFree(weight_in_gpu);
	cudaFree(out_gpu);
}

//�����Ĳ��԰�������
void FullCnnProject() {

	//opencv��ȡͼƬ 512*512
	Mat raw_img = imread("Lena.jpg");
	Mat grey_img,small_img;
	printf("ͼƬ��СΪ %d��  %d��  %dͨ�� \n ", raw_img.rows, raw_img.cols, raw_img.channels());
	cout << "��������Ϊ��"<< raw_img.type() << endl;
	//imshow("��ȡͼ��չʾ", img);
	//waitKey(1500);
	//ת���ɻҶ�ͼ
	//cvtColor(raw_img, grey_img, CV_BGR2GRAY);
	//printf("ͼƬ��СΪ %d��  %d��  %dͨ�� \n ", grey_img.rows, grey_img.cols, grey_img.channels());

	resize(raw_img, small_img, Size(64, 64), INTER_AREA);
	printf("ͼƬ���ź�Ĵ�СΪ %d��  %d��  %dͨ�� \n ", small_img.rows, small_img.cols, small_img.channels());

	//��ͼ��������ֵת���ɸ�����
	//�����Լ�����float�����飬Ȼ������ת����������
	int pic_input_size = small_img.rows * small_img.cols*raw_img.channels();

	float * img_input=(float *)malloc(pic_input_size *sizeof(float));
	//ѭ�� assignment
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

	//imshow("��Сͼ��չʾ", small_img);
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
		printf(" \n  ͼ��һ��  \n");
	}
	cout << "##########################����Ϊͼ���������########################### " << "\n"<< "\n";

	//////////////////////////////////////////////////////////////////////
	///////        ���������Ǵ��������������������ˣ�����      /////////
	//////////////////////////////////////////////////////////////////////
	
	/////////////////////  һ ��� step1 ////////////////////////////////////
	//���� ԭͼ�������Ѿ�����
	int input_3d_size =small_img.channels()* small_img.rows*small_img.cols;
	int kernel_3d_size = 10 * small_img.channels() * 3*3;
	//step1_out��С
	int step1_out_size = 10 * small_img.channels() * small_img.rows*small_img.cols;

	float *kernel_in_cpu = (float*)malloc(kernel_3d_size * sizeof(float));
	float *feature_out_cpu = (float*)malloc(step1_out_size * sizeof(float));
	//  10�� kernel ��ֵ
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

	//����������
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
		printf(" \n  һ�� ����˵�����  \n ");
	}

	float *feature_in_gpu, *kernel_in_gpu, *out_gpu;
	cudaMalloc((void**)&feature_in_gpu, sizeof(float)*input_3d_size);
	cudaMalloc((void**)&kernel_in_gpu, sizeof(float)*kernel_3d_size);
	cudaMalloc((void**)&out_gpu, sizeof(float)*step1_out_size);

	printf("\n cudaMalloc ���\n");

	// ��CPU�е����ݴ��� GPU�ڴ���
	cudaMemcpy(feature_in_gpu, img_input, sizeof(float)*input_3d_size, cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_in_gpu, kernel_in_cpu, sizeof(float)*kernel_3d_size, cudaMemcpyHostToDevice);

	printf("\n cudaMemcpy ���\n");

	//��ƴ�����߳�  �����߳�
	dim3 dimGrid(2,2,10);
	dim3 dimBlock(32,32);// �̵߳���״

	//����CUDA APIʱ����Ӧ�ô������device���ڴ�ָ�룡����
	//����ֱ�Ӿ͵��ò���kernel
	conv_step1_new<<<dimGrid, dimBlock >>>(out_gpu, feature_in_gpu, kernel_in_gpu,64,3,3);
	printf("\n�����������\n");
	//��GPU����õ��ڴ淵�ظ�CPU
	//copy back
	cudaMemcpy(feature_out_cpu, out_gpu, sizeof(float)*step1_out_size, cudaMemcpyDeviceToHost);

	//���featrue_out_cpu ��� ������
	printf("---------------------------------------\n");
	printf("-------���featrue_out_cpu���-----------\n");
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
			printf(" \n  һ��������Ľ��  \n ");
		}
	}*/
	printf("\n  --------- conv step 1 ȫ����� ------ \n");

	cudaFree(feature_in_gpu);
	cudaFree(kernel_in_gpu);

	//out_gpu���ͷż�������һ��������ֵ

	////////////////////////////////////////////////////////////////////////////////////////
	///////           conv step 2       ���Ǳ�����֮ǰ��  gpu_outָ��            //////////////
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
	printf("-------���conv_step2_cpu���-----------\n");
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
			printf(" \n  ��%d��������Ľ��  \n ", num);
		}
	}
	//����������ͷ���һ���������gpuָ��


	////////////////////////////////////////////////////////////////////
	////////////////////   �ػ� ����   conv_step2_out_gpu ////////////////////////////////
	////////////////////////////////////////////////////////////////////
	//�ػ�����û�� Ȩ�ؾ��������
	int pool_out_size = 8 * 8 * 10;

	float * pool_out_cpu, *pool_out_gpu;

	//�����ڴ�ռ�
	cudaMalloc((void**)&pool_out_gpu, sizeof(float)*pool_out_size);
	pool_out_cpu = (float*)malloc(pool_out_size * sizeof(float));

	//kernel��������
	dim3 gridDim_pool(1, 1, 10);
	dim3 gridBlock_pool(8, 8);
	pool_new<<<gridDim_pool, gridBlock_pool >>>(pool_out_gpu, conv_step2_out_gpu, 64, 8);

	//�������cpu
	cudaMemcpy(pool_out_cpu, pool_out_gpu, sizeof(float)*pool_out_size, cudaMemcpyDeviceToHost);

	for (int num = 0; num < 10; num++) {
		for (int layer = 0; layer < 1; layer++) {
			for (int i = 0; i < 8; i++) {
				for (int j = 0; j < 8; j++) {
					printf("%f \t", pool_out_cpu[num * 64 * 1
						+ i * 8 + j]);
				}
			}
			printf(" \n  ��%d����Ľ��  \n ", num);
		}
	}
	printf(" \n  �ػ��������гɹ�������NB  \n ");

	// �ͷſ����ͷŵ���ָ��
	cudaFree(conv_step2_out_gpu);

	////////////////////////////////////////////////////////////////////////
	/////////////          ȫ���Ӳ�(��������)      /////////////////////////
	////////////////////////////////////////////////////////////////////////
	//���忪�����ݿռ�Ĵ�С��һЩ�����
	int weight_size = 8 * 8 * 10 * 10;
	int fc_out_size = 10;

	float * fc_out_cpu, *fc_out_gpu,* weight_in_cpu,*weight_in_gpu;

	//���ٶ�Ӧ�Ŀռ�
	fc_out_cpu = (float *)malloc(fc_out_size * sizeof(float));
	weight_in_cpu = (float *)malloc(weight_size * sizeof(float));
	cudaMalloc((void**)&fc_out_gpu, sizeof(float)*fc_out_size);
	cudaMalloc((void**)&weight_in_gpu, sizeof(float)*weight_size);

	//��Ȩ�����鸳��ֵ һ��Ҫ�� mencopy֮ǰ��
	for (int raw = 0; raw < 10; raw++) {
		for (int i = 0; i < 8 * 8 * 10; i++)
		{
			weight_in_cpu[raw * 8 * 8 * 10 + i] = raw *0.1;
		}
	}

	for (int raw = 0; raw < 10; raw++) {
		for (int i = 0; i < 8 * 8 * 10; i++)
		{
			printf(" %f \t",weight_in_cpu[raw * 8 * 8 * 10 + i]);
		}
		printf("\n");
		printf("#########################################");
		printf("########### һ������ ############\n\n");
	}

	cudaMemcpy(weight_in_gpu, weight_in_cpu, sizeof(float)*weight_size, cudaMemcpyHostToDevice);

	//��ƴ��� �߳̽ṹ������kernel����
	dim3 dimGrid_fc(10);
	dim3 dimBlock_fc(1);
	FC_SharedMem <<<dimGrid_fc, dimBlock_fc>>> (pool_out_gpu, weight_in_gpu, fc_out_gpu, 8 * 8 * 10);

	//���������cpu
	cudaMemcpy(fc_out_cpu, fc_out_gpu, sizeof(float)*fc_out_size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < fc_out_size; i++)
	{
		printf("%f \t", fc_out_cpu[i]);
	}
	printf("\n�õ����Ľ��!!!!\n");

}

int main(void){
	FullCnnProject();
}