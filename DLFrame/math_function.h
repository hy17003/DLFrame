#ifndef MATH_FUNCTION
#define MATH_FUNCTION

#include "Blob.h"

namespace hy17003
{

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
void fast_conv(float* input_data, float* kernel, int im_h, int im_w, int k_h, int k_w,
	int pad_h, int pad_w, int stride, float* output_data);
void convolution(Blob bottom, Blob kernel, int pad_h, int pad_w, int stride, Blob& top);

}

#endif