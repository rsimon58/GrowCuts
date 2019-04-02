#include <cstdlib>
#include <ctime>
#include <string>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include "gaussianFilter.h"
#include "helper_cuda.h"
#include "ImageIO.h"
#include "MetalBoundaryCorrection.h"

//using namespace std;

//CUDA function prototypes
static __global__ void d_growcuts(int *d_label, int *d_newLabel, size_t d_labelPitch, float *d_strength, float *d_newStrength, size_t d_strengthPitch,
	float *d_image, size_t d_imagePitch, int *d_converged, int ncols, int nrows);
static __global__ void d_growcuts_Checker(int *d_labels, size_t d_labelPitch, float *d_strength, size_t d_strengthPitch,
	float *d_image, size_t d_imagePitch, int *d_converged, int ncols, int nrows, int d);
int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}


////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrection::initialize()
{
	m_labels = NULL;
	m_newLabels = NULL;
	m_strength = NULL;
	m_newStrength = NULL;

	//cuda
	d_labels = NULL;
	d_newLabels = NULL;
	d_strength = NULL;
	d_newStrength = NULL;

	m_nDilate = 0;
}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrection::dealloc()
{

	if (m_labels)
	{
		delete[] m_labels[0];
		delete[] m_labels;
	}

	if (m_newLabels)
	{
		delete[] m_newLabels[0];
		delete[] m_newLabels;
	}
	if (m_strength)
	{
		delete[] m_strength[0];
		delete[] m_strength;
	}

	if (m_newLabels)
	{
		delete[] m_newStrength[0];
		delete[] m_newStrength;
	}

	if (m_bUsingGPU)
		cleanUpCuda();
}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
MetalBoundaryCorrection::MetalBoundaryCorrection()
{
	initialize();
}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
MetalBoundaryCorrection::~MetalBoundaryCorrection()
{
	dealloc();
}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrection::setup(int rows, int cols, bool bUsingGPU)
{
	m_ncols = cols;
	m_nrows = rows;

	m_bUsingGPU = bUsingGPU;

	if (m_bUsingGPU)
		initializeGPU();
	//else
		initializeCPU();
}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrection::initializeCPU()
{
	int i;


	//allocate memory
	m_labels = new int*[m_nrows];
	m_labels[0] = new int[m_nrows*m_ncols];
	for (i = 1; i < m_nrows; i++)
	{
		m_labels[i] = m_labels[i - 1] + m_ncols;
	}

	m_newLabels = new int*[m_nrows];
	m_newLabels[0] = new int[m_nrows*m_ncols];
	for (i = 1; i < m_nrows; i++)
	{
		m_newLabels[i] = m_newLabels[i - 1] + m_ncols;
	}

	m_strength = new float*[m_nrows];
	m_strength[0] = new float[m_nrows*m_ncols];
	for (i = 1; i < m_nrows; i++)
	{
		m_strength[i] = m_strength[i - 1] + m_ncols;
	}

	m_newStrength = new float*[m_nrows];
	m_newStrength[0] = new float[m_nrows*m_ncols];
	for (i = 1; i < m_nrows; i++)
	{
		m_newStrength[i] = m_newStrength[i - 1] + m_ncols;
	}
}


////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrection::initializeGPU()
{
	//cudaSetDevice( cutGetMaxGflopsDeviceId() );
	cudaSetDevice(0);

	int blocksInX, blocksInY;

	//dimension to configure the computation grids
	m_BlockSize = dim3(16, 16, 1);
	//m_BlockSize = dim3(8, 8, 1);
	blocksInX = iDivUp(m_ncols, m_BlockSize.x);
	blocksInY = iDivUp(m_nrows, m_BlockSize.y);
	m_GridSize = dim3(blocksInX, blocksInY);


	//allocate pitch linear memory for projection image
	cudaMallocPitch((void **)&d_labels, &d_labelPitch, m_ncols*sizeof(int), m_nrows);
	cudaMallocPitch((void **)&d_newLabels, &d_labelPitch, m_ncols*sizeof(int), m_nrows);

	cudaMallocPitch((void **)&d_strength, &d_strengthPitch, m_ncols*sizeof(float), m_nrows);
	cudaMallocPitch((void **)&d_newStrength, &d_strengthPitch, m_ncols*sizeof(float), m_nrows);

	cudaMallocPitch((void **)&d_image, &d_imagePitch, m_ncols*sizeof(float), m_nrows);

	//cudaMallocPitch((void **)&d_mask, &d_maskPitch, m_ncols*sizeof(int), m_nrows);

	cudaMalloc((void **)&d_converged, sizeof(int));

	//Test pushing and pulling d_converged on and off of the device
	//int converged = 1;
	//printf("%d\n", converged);

	//cudaMemcpy((void *)d_converged, &converged, sizeof(int), cudaMemcpyHostToDevice);

	//converged = 0;
	//printf("%d\n", converged);

	//cudaMemcpy((void *)&converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost);
	//printf("%d\n", converged);
}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrection::FindBoundary(float **image, float **mask)
{
	m_mask = mask;
	m_image = image;

	//test();

	setInitalLabelsAndStrength();

	//test();

	if (!m_bUsingGPU)
		growcuts();
	else
		//growcutsGPU()
		growcutsGPU_Checker();

	if (m_nDilate > 0)
		dilateMask(m_nDilate);
}



////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrection::setInitalLabelsAndStrength()
{
	int i, j;

	m_left = m_ncols;
	m_right = 0;
	m_top = m_nrows;
	m_bottom = 0;

	float **maskBlur = new float*[m_nrows];
	maskBlur[0] = new float[m_nrows*m_ncols];
	for (i = 1; i < m_nrows; i++)
	{
		maskBlur[i] = maskBlur[i - 1] + m_ncols;
	}

	memcpy(maskBlur[0], m_mask[0], sizeof(float)*m_nrows*m_ncols);

	GaussianFilter myFilter;
	myFilter.setSigmaX(1.0f);
	myFilter.setSigmaY(1.0f);
	myFilter.setSigmaZ(0.0f);
	myFilter.initializeFilter();
	myFilter.filter(maskBlur, m_nrows, m_ncols);

	#pragma omp parallel for private(j)
	for (i = 1; i < m_nrows-1; i++)
	{
		for ( j = 1; j < m_ncols-1; j++) 
		{
			if (m_mask[i][j] == 0)
			{
				if ( m_mask[i][j - 1] != 0 || m_mask[i][j + 1] != 0 || m_mask[i - 1][j] != 0 || m_mask[i + 1][j] != 0) 
				{
					m_labels[i][j] = -1;       
					m_strength[i][j] = 1.0f;

					if (j < m_left)
						m_left = j;
					if (j > m_right)
						m_right = j;

					if (i < m_top)
						m_top = i;

					if (i > m_bottom)
						m_bottom = i;
				}
				else
				{
					m_labels[i][j] = -999;
					m_strength[i][j] = 1.0f;
				}
			}
			else if(m_mask[i][j] == 1)
			{
				if (m_mask[i][j - 1] != 1 || m_mask[i][j + 1] != 1 || m_mask[i - 1][j] != 1 || m_mask[i + 1][j] != 1)
				{
					m_labels[i][j] = 1;
					m_strength[i][j] = 1.0f;
				}
				else
				{
					m_labels[i][j] = 999;
					m_strength[i][j] = 1.0f;
				}
			}
			else
			{
				m_labels[i][j] = 0;
				m_strength[i][j] = 0.0f;
				//if (m_mask[i][j] < 0.5f)
				//{
				//	m_labels[i][j] = -1;
				//	m_strength[i][j] = 0.9f;//1.0f
				//}
				//else
				//{
				//	m_labels[i][j] = 0;
				//	m_strength[i][j] = 0.0f;
				//}
			}

			//add internal background seed points
			if ((maskBlur[i][j] - m_mask[i][j]) >= 0.03f)
			{
				m_labels[i][j] = -1;
				m_strength[i][j] = 1.0f;
			}
		}
	}

	delete[] maskBlur[0];
	delete[] maskBlur;
}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrection::dilateMask(int nDilate)
{
	int i, j;

	for(int k = 0;  k < nDilate; k++)
	{
		memcpy(m_strength[0], m_mask[0], sizeof(float)*m_nrows*m_ncols);

		for (i = m_top-1; i < m_bottom+1; i++)
		{
			for (j = m_left-1; j < m_right+1; j++)
			{
				if (m_mask[i][j] == 0)
				{
					if ( m_mask[i][j - 1] != 0 || m_mask[i][j + 1] != 0 || m_mask[i - 1][j] != 0 || m_mask[i + 1][j] != 0) 
					{
						m_strength[i][j] = 1.0f;
					}
				}
			}
		}

		memcpy(m_mask[0], m_strength[0], sizeof(float)*m_nrows*m_ncols);
	}

}


////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrection::growcuts()
{

	int i, j,  m;
	int ii, jj;
	//int Nx[] = { -1, 1,  0, 0, -1, -1, 1,  1 }; //8-neighbors
	//int Ny[] = {  0, 0, -1, 1,  1, -1, 1, -1 };
	//int nNeighbors = 8;
	int Nx[] = { -1, 1,  0, 0 }; //4-neighbors
	int Ny[] = {  0, 0, -1, 1 };
	int nNeighbors = 4;
	float del;
	float C, g;
	float maxC = 7000.0f;
	int converged;
	int MAX_ITS = 100;
	int its = 0;
	int cnt;
	float max = 0.0f;

	memcpy(m_newLabels[0], m_labels[0], sizeof(int)*m_nrows*m_ncols);
	memcpy(m_newStrength[0], m_strength[0], sizeof(float)*m_nrows*m_ncols);

	converged = 0;
	while (!converged)
	{
		its++;
		converged = 1; 

		cnt = 0;
		#pragma omp parallel for private(j)
		//for every pixel p
		for (i = m_top-1; i < m_bottom+1; i++)
		{
			for (j = m_left-1; j < m_right+1; j++)
			{
				//these pixels are frozen
				if (m_labels[i][j] == 999 || m_labels[i][j] == -999)
					continue;


				//for every neighbor q
				for (m = 0; m < nNeighbors; m++)
				{

					ii = i + Ny[m];
					jj = j + Nx[m];

					del = m_image[i][j] - m_image[ii][jj];
					C = sqrt(del*del);

					g = 1 - (C / maxC); //attack force

					if (C > max)
						max = C;

					g *= m_strength[ii][jj];

					if (g > m_strength[i][j]) //attack succeeds
					{
						m_newStrength[i][j] = g;
						m_newLabels[i][j] = m_labels[ii][jj];
						converged = 0; // keep iterating
						cnt++;
					}

				}

			}
		}

		//copy prev result
		memcpy(m_labels[0], m_newLabels[0], sizeof(int)*m_nrows*m_ncols);
		memcpy(m_strength[0], m_newStrength[0], sizeof(float)*m_nrows*m_ncols);

		//check cnt
		if (its == MAX_ITS)
			break;

		//printf("count %d\n",cnt);
	}

	//generate new mask
	for (i = 0; i < m_nrows; i++)
	{
		for (j = 0; j < m_ncols; j++)
		{
			if (m_newLabels[i][j] == 0)
				int mm = 0;
			if (m_newLabels[i][j] == -1 || m_newLabels[i][j] == 0 || m_newLabels[i][j] == -999)
				m_mask[i][j] = 0.0f;
			else
				m_mask[i][j] = 1.0f;
		}
	}

	printf("total iterations=%d\n",its);
	printf("C %f\n", max);



}
////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrection::growcutsGPU()
{

	int i, j;
	int converged;
	int MAX_ITS = 100;
	int its = 0;
	//int cnt;
	float max = 0.0f;
	cudaError_t cudaStatus;

	//copy image to device
	cudaStatus = cudaMemcpy2D((void *)d_image, d_imagePitch, (void *)&m_image[0][0], m_ncols*sizeof(float), m_ncols*sizeof(float), m_nrows, cudaMemcpyHostToDevice);

	//copy initial labels to device
	cudaStatus = cudaMemcpy2D((void *)d_labels, d_labelPitch, (void *)&m_labels[0][0], m_ncols*sizeof(int), m_ncols*sizeof(int), m_nrows, cudaMemcpyHostToDevice);
	//cudaStatus = cudaMemcpy2D((void *)d_newLabels, d_labelPitch, (void *)d_labels, d_labelPitch, m_ncols*sizeof(int), m_nrows, cudaMemcpyDeviceToDevice);
	cudaStatus = cudaMemcpy2D((void *)d_newLabels, d_labelPitch, (void *)d_labels, d_labelPitch, d_labelPitch, m_nrows, cudaMemcpyDeviceToDevice);

	//copy initial strength to device
	cudaStatus = cudaMemcpy2D((void *)d_strength, d_strengthPitch, (void *)&m_strength[0][0], m_ncols*sizeof(float), m_ncols*sizeof(float), m_nrows, cudaMemcpyHostToDevice);
	//cudaStatus = cudaMemcpy2D((void *)d_newStrength, d_strengthPitch, (void *)d_strength, d_strengthPitch, m_ncols*sizeof(float), m_nrows, cudaMemcpyDeviceToDevice);
	cudaStatus = cudaMemcpy2D((void *)d_newStrength, d_strengthPitch, (void *)d_strength, d_strengthPitch, d_strengthPitch, m_nrows, cudaMemcpyDeviceToDevice);

	converged = 0;
	while (!converged)
	{
		its++;

		converged = 1;
		cudaMemcpy((void *)d_converged, &converged, sizeof(int), cudaMemcpyHostToDevice);

		d_growcuts <<<m_GridSize, m_BlockSize >>>(d_labels, d_newLabels, d_labelPitch / sizeof(int), d_strength, d_newStrength, d_strengthPitch / sizeof(float),
			d_image, d_imagePitch / sizeof(float), d_converged, m_ncols, m_nrows);

		cudaThreadSynchronize();
		getLastCudaError("d_growcuts kernel failed");

		//copy prev result
		//cudaStatus = cudaMemcpy2D((void *)d_labels, d_labelPitch, (void *)d_newLabels, d_labelPitch, m_ncols*sizeof(int), m_nrows, cudaMemcpyDeviceToDevice);
		//cudaStatus = cudaMemcpy2D((void *)d_strength, d_strengthPitch, (void *)d_newStrength, d_strengthPitch, m_ncols*sizeof(float), m_nrows, cudaMemcpyDeviceToDevice);

		cudaStatus = cudaMemcpy2D((void *)d_labels, d_labelPitch, (void *)d_newLabels, d_labelPitch, d_labelPitch, m_nrows, cudaMemcpyDeviceToDevice);
		cudaStatus = cudaMemcpy2D((void *)d_strength, d_strengthPitch, (void *)d_newStrength, d_strengthPitch, d_strengthPitch, m_nrows, cudaMemcpyDeviceToDevice);

		cudaMemcpy((void *)&converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost);

		//printf("converged %d\n", converged);
		//check cnt
		if (its == MAX_ITS)
			break;

		//printf("count %d\n",cnt);
	}

	//copy label from device to host
	cudaStatus = cudaMemcpy2D((void *)&m_newLabels[0][0], m_ncols*sizeof(int), (void *)d_newLabels, d_labelPitch, m_ncols*sizeof(int), m_nrows, cudaMemcpyDeviceToHost);

	//generate new mask
	for (i = 0; i < m_nrows; i++)
	{
		for (j = 0; j < m_ncols; j++)
		{
			if (m_newLabels[i][j] == 0)
				int mm = 0;
			if (m_newLabels[i][j] == -1 || m_newLabels[i][j] == 0 || m_newLabels[i][j] == -999)
				m_mask[i][j] = 0.0f;
			else
				m_mask[i][j] = 1.0f;
		}
	}

	printf("total iterations=%d\n", its);
	printf("C %f\n", max);



}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrection::growcutsGPU_Checker()
{

	int i, j;
	int converged;
	int MAX_ITS = 100;
	int its = 0;
	//int cnt;
	float max = 0.0f;
	cudaError_t cudaStatus;

	//copy image 
	cudaStatus = cudaMemcpy2D((void *)d_image, d_imagePitch, (void *)&m_image[0][0], m_ncols*sizeof(float), m_ncols*sizeof(float), m_nrows, cudaMemcpyHostToDevice);

	//copy initial labels to device
	cudaStatus = cudaMemcpy2D((void *)d_labels, d_labelPitch, (void *)&m_labels[0][0], m_ncols*sizeof(int), m_ncols*sizeof(int), m_nrows, cudaMemcpyHostToDevice);
	//cudaStatus = cudaMemcpy2D((void *)d_newLabels, d_labelPitch, (void *)d_labels, d_labelPitch, d_labelPitch, m_nrows, cudaMemcpyDeviceToDevice);

	//copy initial strength to device
	cudaStatus = cudaMemcpy2D((void *)d_strength, d_strengthPitch, (void *)&m_strength[0][0], m_ncols*sizeof(float), m_ncols*sizeof(float), m_nrows, cudaMemcpyHostToDevice);
	//cudaStatus = cudaMemcpy2D((void *)d_newStrength, d_strengthPitch, (void *)d_strength, d_strengthPitch, d_strengthPitch, m_nrows, cudaMemcpyDeviceToDevice);

	converged = 0;
	while (!converged)
	{
		its++;

		converged = 1;
		cudaMemcpy((void *)d_converged, &converged, sizeof(int), cudaMemcpyHostToDevice);

		d_growcuts_Checker << <m_GridSize, m_BlockSize >> >(d_labels, d_labelPitch / sizeof(int), d_strength, d_strengthPitch / sizeof(float),
			d_image, d_imagePitch / sizeof(float), d_converged, m_ncols, m_nrows, 0);
		cudaThreadSynchronize();

		d_growcuts_Checker << <m_GridSize, m_BlockSize >> >(d_labels, d_labelPitch / sizeof(int), d_strength, d_strengthPitch / sizeof(float),
			d_image, d_imagePitch / sizeof(float), d_converged, m_ncols, m_nrows, 1);
		cudaThreadSynchronize();

		cudaMemcpy((void *)&converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost);

		//printf("converged %d\n", converged);
		//check cnt
		if (its == MAX_ITS)
			break;

		//printf("count %d\n",cnt);
	}

	//copy label from device to host
	cudaStatus = cudaMemcpy2D((void *)&m_labels[0][0], m_ncols*sizeof(int), (void *)d_labels, d_labelPitch, m_ncols*sizeof(int), m_nrows, cudaMemcpyDeviceToHost);

	//generate new mask
	for (i = 0; i < m_nrows; i++)
	{
		for (j = 0; j < m_ncols; j++)
		{
			if (m_labels[i][j] == 0)
				int mm = 0;
			if (m_labels[i][j] == -1 || m_labels[i][j] == 0 || m_labels[i][j] == -999)
				m_mask[i][j] = 0.0f;
			else
				m_mask[i][j] = 1.0f;
		}
	}
	printf("total iterations=%d\n", its);
	printf("C %f\n", max);



}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrection::cleanUpCuda()
{
		cudaFree(d_labels);
		cudaFree(d_newLabels);
		cudaFree(d_newStrength);
		cudaFree(d_strength);
}


////////////////////////////////////////////////////////////////////////
//gpu implementation of growcuts
////////////////////////////////////////////////////////////////////////
static __global__ void d_growcuts(int *d_labels, int *d_newLabels, size_t d_labelPitch, float *d_strength, float *d_newStrength, size_t d_strengthPitch,
	float *d_image, size_t d_imagePitch, int *d_converged, int ncols, int nrows)

{

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	//if (j >= ncols || i >= nrows)
	//	return;

	if (j < 1 || i < 1 || j >= ncols-1 || i >= nrows-1)
		return;

	if (d_labels[i * d_labelPitch + j] == 999 || d_labels[i* d_labelPitch + j] == -999)
		return;       //pixel is frozen

	float del, g, C;
	float maxC = 7000.0f;
	int ii, jj;
	int Nx[] = { -1, 1, 0, 0 }; //4-neighbors
	int Ny[] = { 0, 0, -1, 1 };
	int nNeighbors = 4;

	//for every neighbor q
	for (int m = 0; m < nNeighbors; m++)
	{

		ii = i + Ny[m];
		jj = j + Nx[m];

		del = d_image[i * d_imagePitch + j] - d_image[ii * d_imagePitch + jj];

		C = sqrt(del*del);

		g = 1 - (C / maxC); //attack force

		//if (C > max)
		//	max = C;

		g *= d_strength[ii * d_strengthPitch + jj];

		if (g > d_strength[i * d_strengthPitch + j]) //attack succeeds
		{
			d_newStrength[i * d_strengthPitch + j] = g;
			d_newLabels[i * d_labelPitch + j] = d_labels[ii * d_labelPitch + jj];
			*d_converged = 0; // keep iterating
		}

	}

}

///////////////////////////////////////////////////////////////////////////////////////////
//Checker board approach 
///////////////////////////////////////////////////////////////////////////////////////////
static __global__ void d_growcuts_Checker(int *d_labels,  size_t d_labelPitch, float *d_strength, size_t d_strengthPitch,
	float *d_image, size_t d_imagePitch, int *d_converged, int ncols, int nrows, int d)

{

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int bOdd;

	if (y & 1)
		bOdd = 1; // row i is odd
	else
		bOdd = 0;

	//flip logic
	if (d == 1)
		bOdd = !bOdd;

	//center of block
	const int j = x * 2.0f + bOdd;
	const int i = y;

	if (j < 1 || i < 1 || j >= ncols - 1 || i >= nrows - 1)
		return;

	/*if (d_labels[i * d_labelPitch + j] == 999 || d_labels[i* d_labelPitch + j] == -999)
		return;*/

	if (d_strength[i * d_labelPitch + j] == 1.0f)
		return;

	float del, g, C;
	float maxC = 7000.0f;
	int ii, jj;
	int Nx[] = { -1, 1, 0, 0 }; //4-neighbors
	int Ny[] = { 0, 0, -1, 1 };
	int nNeighbors = 4;

	//for every neighbor q
	for (int m = 0; m < nNeighbors; m++)
	{

		ii = i + Ny[m];
		jj = j + Nx[m];

		del = d_image[i * d_imagePitch + j] - d_image[ii * d_imagePitch + jj];

		C = sqrt(del*del);

		g = 1 - (C / maxC); //attack force

		//if (C > max)
		//	max = C;

		g *= d_strength[ii * d_strengthPitch + jj];

		if (g > d_strength[i * d_strengthPitch + j]) //attack succeeds
		{
			d_strength[i * d_strengthPitch + j] = g;
			d_labels[i * d_labelPitch + j] = d_labels[ii * d_labelPitch + jj];
			*d_converged = 0; // keep iterating
		}

	}

}