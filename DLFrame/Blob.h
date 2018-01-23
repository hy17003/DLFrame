#ifndef BLOB_H
#define BLOB_H

#include <assert.h>
#include <vector>
#include <iostream>

using namespace std;

namespace hy17003 {

class Blob
{
public:
	Blob();
	//���ƹ��캯��
	Blob(Blob& blob_);
	//����=�Ų�����
	Blob& operator =(Blob& blob_);
	//����[]�Ų�����
	float& operator[](int n);
	~Blob();
	vector<int> GetShape();
	int GetCount();
	int GetBatch();
	int GetChannel();
	int GetHeight();
	int GetWidth();
	void Create(vector<int> shape, float value = 0);
	void Create(int n, int c, int h, int w, float value = 0);
	float GetElement(int n, int c, int h, int w);
	void SetElement(float data, int n, int c, int h, int w);
	float* GetBuffer();
	void Reshape(int n, int c, int h, int w);
	void Reshape(vector<int> shape);
	void SetBlob(float* data, int n, int c, int h, int w);
	void SetBlob(float* data, vector<int> shape);
	void Print();
	void Transpose();
private:
	vector<int> shape_;
	float* buffer_;
};

}

#endif