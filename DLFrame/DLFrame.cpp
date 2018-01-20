// DLFrame.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include "Blob.h"
#include "math_function.h"

using namespace std;
using namespace hy17003;

void main()
{
	float kernel_data[54] = {
		/*batch = 0*/
		0.36, -0.40, -0.01,
		-0.13, 0.23, 0.31,
		0.30, 0.26, 0.11,

		-0.12, 0.10, -0.32,
		-0.32, 0.03, -0.11,
		0.20, -0.17, -0.21,

		/*batch = 1*/
		-0.23, -0.23, -0.35,
		0.03, -0.21, 0.31,
		0.19, 0.41, -0.01,

		-0.13, 0.23, -0.31,
		0.35, 0.40, -0.29,
		0.31, 0.25, -0.00,

		/*batch = 2*/
		0.11, -0.09, -0.38,
		-0.05, -0.15, 0.41,
		-0.08, -0.06, 0.12,

		-0.00, 0.39, 0.40,
		-0.19, -0.09, -0.34,
		0.26, 0.33, 0.23
	};

	float bottom_data[288] =
	{
		0.00, 0.00, 0.00, 0.00, 0.02, 0.01, 0.13, 0.14, 0.32, 0.18, 0.25, 0.12,
		0.00, 0.05, 0.22, 0.46, 0.51, 0.47, 0.57, 0.43, 0.54, 0.61, -0.01, 0.16,
		0.00, 0.04, 0.41, 0.11, 0.16, -0.02, 0.16, -0.10, 0.72, 0.61, 0.78, 0.41,
		0.00, 0.06, 0.29, 0.83, 1.27, 1.00, 1.41, 1.02, 1.08, 0.92, 0.72, 0.28,
		0.00, 0.02, 0.37, 0.81, 1.24, 1.00, 1.32, 0.64, 0.76, 0.00, 0.00, 0.00,
		0.00, 0.00, 0.02, 0.22, 0.40, 0.87, 0.82, 0.27, 0.23, 0.06, 0.00, 0.00,
		0.00, 0.00, 0.00, 0.01, 0.38, 0.36, 0.76, 0.71, 0.22, 0.26, 0.01, 0.00,
		0.00, 0.00, 0.00, 0.00, 0.12, 0.56, 0.88, 1.05, 0.62, 0.44, 0.06, 0.00,
		0.00, 0.00, 0.05, 0.24, 0.39, 0.44, 0.42, 0.20, 0.89, 0.77, 0.07, 0.00,
		0.14, 0.31, 0.56, 0.49, 0.24, 0.03, 0.87, 1.06, 0.88, 0.62, 0.00, 0.00,
		0.31, 0.35, 0.07, 0.06, 0.58, 1.07, 1.08, 0.98, 0.53, 0.08, 0.00, 0.00,
		0.16, 0.51, 1.03, 1.22, 1.10, 0.98, 0.52, 0.10, 0.00, 0.00, 0.00, 0.00,

		0.00, 0.00, 0.00, 0.00, 0.02, 0.03, 0.26, 0.18, 0.51, 0.52, 0.11, 0.27,
		0.00, 0.04, 0.41, 0.21, 0.42, 0.40, 0.41, 0.38, 0.57, 0.22, 0.38, 0.31,
		0.00, 0.03, 0.22, 0.06, 0.29, 0.11, 0.05, -0.34, -0.19, -0.40, -0.33, -0.06,
		0.00, 0.00, -0.17, -0.49, -0.70, -0.91, -0.63, -0.49, -0.50, -0.08, -0.05, 0.00,
		0.00, 0.00, -0.08, -0.23, -0.53, -0.17, -0.26, 0.25, 0.16, 0.00, 0.00, 0.00,
		0.00, 0.00, 0.00, 0.00, -0.29, 0.06, -0.07, 0.24, 0.26, 0.18, 0.00, 0.00,
		0.00, 0.00, 0.00, 0.00, -0.04, -0.01, -0.24, -0.20, 0.11, 0.30, 0.08, 0.00,
		0.00, 0.00, 0.00, 0.00, 0.17, 0.29, -0.07, 0.02, -0.42, 0.28, 0.01, 0.00,
		0.00, 0.00, 0.08, 0.39, 0.33, 0.21, 0.15, -0.51, -0.25, -0.36, 0.00, 0.00,
		0.21, 0.34, 0.38, 0.30, 0.22, -0.13, -0.60, -0.75, -0.23, -0.23, 0.00, 0.00,
		0.35, 0.20, 0.16, -0.14, -0.75, -0.61, -0.30, -0.21, -0.00, 0.00, 0.00, 0.00,
		-0.08, -0.16, -0.53, -0.74, -0.32, -0.22, -0.01, 0.00, 0.00, 0.00, 0.00, 0.00
	};

	Blob kernel, bottom, top;
	kernel.SetBlob(kernel_data, 3, 2, 3, 3);
	bottom.SetBlob(bottom_data, 1, 2, 12, 12);
	convolution(bottom, kernel, 0, 0, 1, top);
	top.Print();
	system("pause");
}