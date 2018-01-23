#include "stdafx.h"
#include "define.h"
#include "Blob.h"


namespace hy17003{


Blob::Blob() :buffer_(0)
{
	shape_ = vector<int>(4, 0);
}

//复制构造函数

Blob::Blob(Blob& blob_) 
	:buffer_(0)
{
	shape_ = vector<int>(4, 0);
	SetBlob(blob_.GetBuffer(), blob_.GetShape());
}

//重载=号操作符

Blob& Blob::operator = (Blob& blob_)
{
	if (this != &blob_)
	{
		SetBlob(blob_.GetBuffer(), blob_.GetShape());
	}
	return *this;
}

float& Blob::operator[](int n)
{
	assert(n < GetCount() && n >= 0);
	float& ref = buffer_[n];
	return ref;
}

Blob::~Blob()
{
	shape_ = vector<int>(4, 0);
	if (buffer_ != 0)
	{
		delete[] buffer_;
	}
}


vector<int> Blob::GetShape()
{
	return shape_;
}


float Blob::GetElement(int n, int c, int h, int w)
{
	int row_size = shape_[3];
	int channel_size = shape_[2] * row_size;
	int batch_size = shape_[1] * channel_size;
	int idx = n * batch_size + c * channel_size + h * row_size + w;
	assert(idx < GetCount());
	return buffer_[idx];
}


void Blob::SetElement(float data, int n, int c, int h, int w)
{
	int idx = n * c * h * w;
	assert(idx < GetCount());
	buffer_[idx] = data;
}


void Blob::Reshape(int n, int c, int h, int w)
{
	assert(n * c * h * w == GetCount());
	shape_[0] = n;
	shape_[1] = c;
	shape_[2] = h;
	shape_[3] = w;
}


void Blob::Reshape(vector<int> shape)
{
	assert(shape.size() == 4);
	shape_ = shape;
}


void Blob::SetBlob(float* data, int n, int c, int h, int w)
{
	assert(n > 0 && c > 0 && h > 0 && w > 0);
	if (buffer_ != 0)
		delete[] buffer_;
	int count = n * c * h * w;
	buffer_ = new float[count];
	memcpy(buffer_, data, count * sizeof(float));
	shape_[0] = n;
	shape_[1] = c;
	shape_[2] = h;
	shape_[3] = w;
}


void Blob::SetBlob(float* data, vector<int> shape)
{
	assert(shape.size() == 4);
	SetBlob(data, shape[0], shape[1], shape[2], shape[3]);
}


float* Blob::GetBuffer()
{
	return buffer_;
}


int Blob::GetCount()
{
	int count = 1;
	for (int i = 0; i < shape_.size(); i++)
	{
		count *= shape_[i];
	}
	return count;
}


void Blob::Print()
{
	vector<int> blob_shape = GetShape();
	if (blob_shape.size() != 4)
		return;
	for (int n = 0; n < blob_shape[0]; n++)
	{
		printf("\n /*axis = batch %d*/\n", n);
		for (int c = 0; c < blob_shape[1]; c++)
		{
			printf("\n /*axis = channel %d*/\n", c);
			for (int h = 0; h < blob_shape[2]; h++)
			{
				for (int w = 0; w < blob_shape[3]; w++)
				{
					printf("%.03f ", GetElement(n, c, h, w));
				}
				printf("\n");
			}
		}
	}
}

void Blob::Create(vector<int> shape, float value)
{
	assert(shape.size() == 4);
	Create(shape[0], shape[1], shape[2], shape[3]);
}

void Blob::Create(int n, int c, int h, int w, float value)
{
	assert(n > 0 && c > 0 && h > 0 && w > 0);
	shape_[0] = n;
	shape_[1] = c;
	shape_[2] = h;
	shape_[3] = w;
	int count = n * c * h * w;
	if (buffer_ != 0)
	{
		delete[] buffer_;
	}
	buffer_ = new float[count];
	if (value == 0)
	{
		memset(buffer_, value, sizeof(float) * count);
	}
	else
	{
		for (int i = 0; i < count; i++)
		{
			buffer_[i] = value;
		}
	}
	
}

void Blob::Transpose()
{
	float* new_buffer = new float[GetCount()];
	memset(new_buffer, 0, sizeof(float) * GetCount());
	vector<int> new_shape(4, 0);
	new_shape[0] = shape_[0];
	new_shape[1] = shape_[1];
	new_shape[2] = shape_[3];
	new_shape[3] = shape_[2];
	int channel_size = shape_[2] * shape_[3];
	int batch_size = shape_[1] * channel_size;
	int src_h = shape_[2];
	int src_w = shape_[3];
	int dst_h = src_w;
	int dst_w = src_h;
	for (int n = 0; n < GetBatch(); n++)
	{
		for (int c = 0; c < GetChannel(); c++)
		{
			for (int i = 0; i < new_shape[2]; i++)
			{
				for (int j = 0; j < new_shape[3]; j++)
				{
					int src_idx = n * batch_size + c * channel_size + j * src_w + i;
					int dst_idx = n * batch_size + c * channel_size + i * dst_w + j;
					new_buffer[dst_idx] = buffer_[src_idx];
				}
			}
		}
	}
	delete[] buffer_;
	buffer_ = new_buffer;
	shape_ = new_shape;
}

int Blob::GetBatch()
{
	return shape_[0];
}

int Blob::GetChannel()
{
	return shape_[1];
}

int Blob::GetHeight()
{
	return shape_[2];
}

int Blob::GetWidth()
{
	return shape_[3];
}


}

