#include "stdafx.h"
#include "math_function.h"

namespace hy17003
{

void padMatrix(float* data, int row, int col, int pad_h, int pad_w, float* outData)
{
	int out_h = row + 2 * pad_h;
	int out_w = col + 2 * pad_w;
	for (int i = 0; i < out_h; i++)
	{
		for (int j = 0; j < out_w; j++)
		{
			float& pix = outData[i * out_w + j];
			if ((i < pad_h || i >= out_h - pad_h || j < pad_w || j >= out_w - pad_w))
			{
				pix = 0;
			}
			else
			{
				pix = data[(i - pad_h) * col + j - pad_w];
			}
		}
	}

}


void addMatrix(float* data1, float* data2, int row, int col, float* outData)
{
	int size = row * col;
	for (int i = 0; i < size; i++)
	{
		outData[i] = data1[i] + data2[i];
	}
}

bool is_a_ge_zero_and_a_lt_b(int a, int b) {
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}


void im2col_cpu(float* data_im, int channels,
	int height, int width, int kernel_h, int kernel_w,
	int pad_h, int pad_w,
	int stride_h, int stride_w,
	float* data_col) {
	int output_h = (height + 2 * pad_h -
		((kernel_h - 1) + 1)) / stride_h + 1;
	int output_w = (width + 2 * pad_w -
		((kernel_w - 1) + 1)) / stride_w + 1;
	int channel_size = height * width;
	for (int channel = channels; channel--; data_im += channel_size) {
		for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				int input_row = -pad_h + kernel_row;
				for (int output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						for (int output_cols = output_w; output_cols; output_cols--) {
							*(data_col++) = 0;
						}
					}
					else {
						int input_col = -pad_w + kernel_col;
						for (int output_col = output_w; output_col; output_col--) {
							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
								*(data_col++) = data_im[input_row * width + input_col];
							}
							else {
								*(data_col++) = 0;
							}
							input_col += stride_w;
						}
					}
					input_row += stride_h;
				}
			}
		}
	}
}



void conv_nopad(float* input_data, float* kernel, int im_h, int im_w, int k_h, int k_w, int stride,
	float* output_data)
{
	int out_h = (im_h - k_h) / stride + 1;
	int out_w = (im_w - k_w) / stride + 1;
	memset(output_data, 0, sizeof(float) * out_h * out_w);
	for (int i = 0; i < out_h; i++)
	{
		for (int j = 0; j < out_w; j++)
		{
			//对应原图中的位置
			int src_i = i * stride + 0.5 * k_h;
			int src_j = j * stride + 0.5 * k_w;
			for (int k_i = -k_h / 2; k_i <= k_h / 2; k_i++)
			{
				for (int k_j = -k_w / 2; k_j <= k_w / 2; k_j++)
				{
					int k_idx = int((k_i + k_h / 2) * k_w + k_j + k_w / 2);
					float& k_d = kernel[k_idx];
					float& s_d = input_data[(src_i + k_i) * im_w + src_j + k_j];
					output_data[i * out_w + j] += k_d * s_d;
				}
			}
		}
	}
}


void dot(float* m1, int m1_h, int m1_w, float* m2, int m2_h, int m2_w, float* mr)
{
	int mr_h = m1_h;
	int mr_w = m2_w;
	assert(m1_w == m2_h);
	for (int i = 0; i < mr_h; i++)
	{
		for (int j = 0; j < mr_w; j++)
		{
			//结果的第i行第j列是m1的第i行与m2的第j列的内积
			mr[i * mr_w + j] = 0;
			for (int ii = 0; ii < m1_w; ii++)
			{
				mr[i * mr_w + j] += m1[i * m1_w + ii] * m2[ii * m2_w + j];
			}
		}
	}
}


void fast_conv(float* input_data, float* kernel, int im_h, int im_w, int k_h, int k_w,
	int pad_h, int pad_w, int stride, float* output_data)
{
	//输出大小
	int out_h = (im_h + 2 * pad_h - k_h) / stride + 1;
	int out_w = (im_w + 2 * pad_w - k_w) / stride + 1;
	//数据矩阵转换后矩阵大小
	int c_h = k_h * k_w;
	int c_w = out_h * out_w;
	//生成转换后的数据矩阵
	float *conver_data = new float[c_h * c_w];
	im2col_cpu(input_data, 1, im_h, im_w, k_h, k_w, pad_h, pad_w, stride, stride, conver_data);
	//将数据矩阵与核矩阵点乘
	dot(kernel, 1, k_w * k_h, conver_data, c_h, c_w, output_data);
	//将结果转成正常的图像矩阵
	delete[] conver_data;
	conver_data = 0;
}

void convolution(Blob bottom, Blob kernel, int pad_h, int pad_w, int stride, Blob& top)
{
	vector<int> bottom_shape = bottom.GetShape();
	vector<int> kernel_shape = kernel.GetShape();
	assert(bottom_shape[1] == kernel_shape[1]);
	//计算输出的形状
	vector<int> top_shape(4, 0);
	top_shape[0] = bottom_shape[0];
	top_shape[1] = kernel_shape[0];
	top_shape[2] = (bottom_shape[2] + 2 * pad_h - kernel_shape[2]) / stride + 1;
	top_shape[3] = (bottom_shape[3] + 2 * pad_w - kernel_shape[3]) / stride + 1;
	top.Create(top_shape);
	//计算各偏移量
	int top_channel_count = top_shape[2] * top_shape[3];
	int top_batch_count = top_shape[1] * top_channel_count;
	int bottom_channel_count = bottom_shape[2] * bottom_shape[3];
	int bottom_batch_count = bottom_shape[1] * bottom_channel_count;
	int kernel_channel_count = kernel_shape[2] * kernel_shape[3];
	int kernel_batch_count = kernel_shape[1] * kernel_channel_count;
	//计算卷积
	for (int n = 0; n < top_shape[0]; n++)
	{
		//分别计算各channel
		for (int c = 0; c < top_shape[1]; c++)
		{
			//每一个channel是将核与输入图像对应通道卷积后相加
			float *single_top = top.GetBuffer() + (top_batch_count * n + c * top_channel_count);
			for (int ci = 0; ci < bottom_shape[1]; ci++)
			{
				float *single_bottom = bottom.GetBuffer() + (bottom_batch_count * n + ci * bottom_channel_count);
				float *single_kernel = kernel.GetBuffer() + (kernel_batch_count * c + ci * kernel_channel_count);
				//为fast_conv输出结果tmp_top分类内存
				float *tmp_top = 0;
				tmp_top = new float[top_channel_count];
				//快速卷积
				fast_conv(single_bottom, single_kernel, bottom_shape[2], bottom_shape[3], kernel_shape[2],
					kernel_shape[3], pad_h, pad_w, stride, tmp_top);
				addMatrix(single_top, tmp_top, top.GetHeight(), top.GetWidth(), single_top);
				delete[] tmp_top;
			}
		}
	}
}

}