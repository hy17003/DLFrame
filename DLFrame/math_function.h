#ifndef MATH_FUNCTION
#define MATH_FUNCTION
#include "define.h"
#include <math.h>
#include "Blob.h"




namespace hy17003
{
	enum POOL_TYPE
	{
		POOL_MAX = 0,
		POOL_MEAN = 1
	};

void printMatrix(float* data, int row, int col, int row_start = 0, int row_end = -1, int col_start = 0, int col_end = -1);
void padMatrix(float* data, int row, int col, int pad_h, int pad_w, float* outData);
void addMatrix(float* data1, float* data2, int row, int col, float* outData);
bool is_a_ge_zero_and_a_lt_b(int a, int b);
void im2col_cpu(float* data_im, int channels,
	int height, int width, int kernel_h, int kernel_w,
	int pad_h, int pad_w,
	int stride_h, int stride_w,
	float* data_col);
void conv_nopad(float* input_data, float* kernel, int im_h, int im_w, int k_h, int k_w, int stride,
	float* output_data);
void dot(float* m1, int m1_h, int m1_w, float* m2, int m2_h, int m2_w, float* mr);
void blob_dot(Blob matrix_blob, Blob bottom_blob, Blob& top_blob);
void blob_add(Blob input_blob1, Blob input_blob2, Blob& output_blob);
void fast_conv(float* input_data, float* kernel, int im_h, int im_w, int k_h, int k_w,
	int pad_h, int pad_w, int stride, float* output_data);
//卷积
void Convolution(Blob bottom, Blob kernel, int pad_h, int pad_w, int stride, Blob& top);
//池化
void Pooling(Blob bottom, int kernel_h, int kernel_w, int pad_h, int pad_w, int stride, POOL_TYPE type, Blob& top);
//全连接
void InnerProduct(Blob bottom, Blob weight, Blob bias, Blob& top);
//RELU
void Relu(Blob bottom, Blob& top);
//最大序号
vector<int> MaxProbIndex(Blob prob);
}

#endif