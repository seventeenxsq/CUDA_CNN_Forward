
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "Windows.h"
#include <math.h>
#include<iostream>
using namespace std;

#define BLOCKDIM_X		16
#define BLOCKDIM_Y		16

#define GRIDDIM_X		256
#define GRIDDIM_Y		256
#define MASK_WIDTH		5

__constant__ int d_const_Gaussian[MASK_WIDTH * MASK_WIDTH]; //分配常数存储器

unsigned char* readBmp(char* bmpName, int* width, int* height, int* byteCount);
bool saveBmp(char* bmpName, unsigned char* imgBuf, int width, int height, int byteCount);
static __global__ void kernel_GaussianFilt(int width, int height, int byteCount, unsigned char* d_src_imgbuf, unsigned char* d_guassian_imgbuf);

int main()
{
	//查看显卡配置
	struct cudaDeviceProp pror;
	cudaGetDeviceProperties(&pror, 0);
	cout << "maxThreadsPerBlock=" << pror.maxThreadsPerBlock << endl;

	long start, end;
	long time = 0;

	//CUDA计时函数
	start = GetTickCount();
	cudaEvent_t startt, stop; //CUDA计时机制
	cudaEventCreate(&startt);
	cudaEventCreate(&stop);
	cudaEventRecord(startt, 0);

	unsigned char* h_src_imgbuf;  //图像指针
	int width, height, byteCount;
	char rootPath1[] = "D:/测试图片/2.bmp";
	char readPath[1024];
	int frame = 1;
	for (int k = 1; k <= frame; k++)
	{
		sprintf(readPath, "%s%d.bmp", rootPath1, k);
		h_src_imgbuf = readBmp(readPath, &width, &height, &byteCount);

		int size1 = width * height * byteCount * sizeof(unsigned char);
		int size2 = width * height * sizeof(unsigned char);

		//输出图像内存-host端	
		unsigned char* h_guassian_imgbuf = new unsigned char[width * height * byteCount];

		//分配显存空间
		unsigned char* d_src_imgbuf;
		unsigned char* d_guassian_imgbuf;

		cudaMalloc((void**)&d_src_imgbuf, size1);
		cudaMalloc((void**)&d_guassian_imgbuf, size1);

		//把数据从Host传到Device
		cudaMemcpy(d_src_imgbuf, h_src_imgbuf, size1, cudaMemcpyHostToDevice);

		//将高斯模板传入constant memory
		int Gaussian[25] = { 1,4,7,4,1,
							4,16,26,16,4,
							7,26,41,26,7,
							4,16,26,16,4,
							1,4,7,4,1 };//总和为273
		cudaMemcpyToSymbol(d_const_Gaussian, Gaussian, 25 * sizeof(int));

		double bx = ceil((double)width / BLOCKDIM_X); //网格和块的分配
		double by = ceil((double)height / BLOCKDIM_Y);

		if (bx > GRIDDIM_X) bx = GRIDDIM_X;
		if (by > GRIDDIM_Y) by = GRIDDIM_Y;

		dim3 grid(bx, by);//网格的结构
		dim3 block(BLOCKDIM_X, BLOCKDIM_Y);//块的结构

		//kernel--高斯滤波
		kernel_GaussianFilt <<<grid, block>>> (width, height, byteCount, d_src_imgbuf, d_guassian_imgbuf);
		cudaMemcpy(h_guassian_imgbuf, d_guassian_imgbuf, size1, cudaMemcpyDeviceToHost);//数据传回主机端

		char rootPath2[] = "D:\\测试结果\\";
		char writePath[1024];
		sprintf(writePath, "%s%d.bmp", rootPath2, k);
		saveBmp(writePath, h_guassian_imgbuf, width, height, byteCount);

		//输出进度展示
		cout << k << "  " << ((float)k / frame) * 100 << "%" << endl;

		//释放内存
		cudaFree(d_src_imgbuf);
		cudaFree(d_guassian_imgbuf);

		delete[]h_src_imgbuf;
		delete[]h_guassian_imgbuf;
	}
	end = GetTickCount();
	InterlockedExchangeAdd(&time, end - start);
	cout << "Total time GPU:";
	cout << time << endl;
	return 0;
}

static __global__ void kernel_GaussianFilt(int width, int height, int byteCount, unsigned char* d_src_imgbuf, unsigned char* d_dst_imgbuf)
{
	const int tix = blockDim.x * blockIdx.x + threadIdx.x;
	const int tiy = blockDim.y * blockIdx.y + threadIdx.y;

	const int threadTotalX = blockDim.x * gridDim.x;
	const int threadTotalY = blockDim.y * gridDim.y;

	for (int ix = tix; ix < height; ix += threadTotalX)
		for (int iy = tiy; iy < width; iy += threadTotalY)
		{
			for (int k = 0; k < byteCount; k++)
			{
				int sum = 0;//临时值
				int tempPixelValue = 0;
				for (int m = -2; m <= 2; m++)
				{
					for (int n = -2; n <= 2; n++)
					{
						//边界处理，幽灵元素赋值为零
						if (ix + m < 0 || iy + n < 0 || ix + m >= height || iy + n >= width)
							tempPixelValue = 0;
						else
							tempPixelValue = *(d_src_imgbuf + (ix + m) * width * byteCount + (iy + n) * byteCount + k);
						sum += tempPixelValue * d_const_Gaussian[(m + 2) * 5 + n + 2];
					}
				}

				if (sum / 273 < 0)
					*(d_dst_imgbuf + (ix)*width * byteCount + (iy)*byteCount + k) = 0;
				else if (sum / 273 > 255)
					*(d_dst_imgbuf + (ix)*width * byteCount + (iy)*byteCount + k) = 255;
				else
					*(d_dst_imgbuf + (ix)*width * byteCount + (iy)*byteCount + k) = sum / 273;
			}
		}
}

unsigned char* readBmp(char* bmpName, int* width, int* height, int* byteCount)
{
	//打开文件
	FILE* fp = fopen(bmpName, "rb");
	if (fp == 0) return 0;
	//跳过文件头
	fseek(fp, sizeof(BITMAPFILEHEADER), 0);

	//读入信息头
	int w, h, b;
	BITMAPINFOHEADER head;
	fread(&head, sizeof(BITMAPINFOHEADER), 1, fp);
	w = head.biWidth;
	h = head.biHeight;
	b = head.biBitCount / 8;
	int lineByte = (w * b + 3) / 4 * 4; //每行的字节数为4的倍数

	//跳过颜色表 （颜色表的大小为1024）（彩色图像并没有颜色表，不需要这一步）
	if (b == 1)
		fseek(fp, 1024, 1);

	//图像数据
	unsigned char* imgBuf = new unsigned char[w * h * b];
	for (int i = 0; i < h; i++)
	{
		fread(imgBuf + i * w * b, w * b, 1, fp);
		fseek(fp, lineByte - w * b, 1);
	}
	fclose(fp);

	*width = w, * height = h, * byteCount = b;

	return imgBuf;
}


bool saveBmp(char* bmpName, unsigned char* imgBuf, int width, int height, int byteCount)
{
	if (!imgBuf)
		return 0;

	//灰度图像颜色表空间1024，彩色图像没有颜色表
	int palettesize = 0;
	if (byteCount == 1) palettesize = 1024;

	//一行象素字节数为4的倍数
	int lineByte = (width * byteCount + 3) / 4 * 4;

	FILE* fp = fopen(bmpName, "wb");
	if (fp == 0) return 0;

	//填写文件头
	BITMAPFILEHEADER fileHead;
	fileHead.bfType = 0x4D42;
	fileHead.bfSize =
		sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + palettesize + lineByte * height;
	fileHead.bfReserved1 = 0;
	fileHead.bfReserved2 = 0;
	fileHead.bfOffBits = 54 + palettesize;
	fwrite(&fileHead, sizeof(BITMAPFILEHEADER), 1, fp);

	// 填写信息头
	BITMAPINFOHEADER head;
	head.biBitCount = byteCount * 8;
	head.biClrImportant = 0;
	head.biClrUsed = 0;
	head.biCompression = 0;
	head.biHeight = height;
	head.biPlanes = 1;
	head.biSize = 40;
	head.biSizeImage = lineByte * height;
	head.biWidth = width;
	head.biXPelsPerMeter = 0;
	head.biYPelsPerMeter = 0;
	fwrite(&head, sizeof(BITMAPINFOHEADER), 1, fp);

	//颜色表拷贝  
	if (palettesize == 1024)
	{
		unsigned char palette[1024];
		for (int i = 0; i < 256; i++)
		{
			*(palette + i * 4 + 0) = i;
			*(palette + i * 4 + 1) = i;
			*(palette + i * 4 + 2) = i;
			*(palette + i * 4 + 3) = 0;
		}
		fwrite(palette, 1024, 1, fp);
	}

	//准备数据并写文件
	unsigned char* buf = new unsigned char[height * lineByte];
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width * byteCount; j++)
			*(buf + i * lineByte + j) = *(imgBuf + i * width * byteCount + j);
	}
	fwrite(buf, height * lineByte, 1, fp);

	delete[]buf;

	fclose(fp);

	return 1;
}