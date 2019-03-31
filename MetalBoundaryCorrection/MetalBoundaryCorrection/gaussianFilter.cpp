#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <float.h>
#include <math.h>

#include "gaussianFilter.h"
#include "omp.h"


void GaussianFilter::initialize()
{
	m_sigmaX = 1.0f;
	m_sigmaY = 1.0f;
	m_sigmaZ = 1.0f;

	m_wx = NULL;
	m_wy = NULL;
	m_wz = NULL;
}

void GaussianFilter::dealloc()
{
	if(m_wx)
		delete [] m_wx;

	if(m_wy)
		delete [] m_wy;

	if(m_wz)
		delete [] m_wz;
}


GaussianFilter::GaussianFilter()
{
	initialize();
}

GaussianFilter::GaussianFilter(float sigmaX, float sigmaY, float sigmaZ)
{
	initialize();

	m_sigmaX = sigmaX;
	m_sigmaY = sigmaY;
	m_sigmaZ = sigmaZ;

	initializeFilter();
}

GaussianFilter::~GaussianFilter()
{
	dealloc();
}

void GaussianFilter::initializeFilter()
{
	int i;
	float sq2pi = 2.506628f;

	int conv_size = (int) (m_sigmaX*3.0f); // mask is 3 dSigma on each side      
	int nn = 2 * conv_size +1;                                                  
	float s2 = -0.5f / (m_sigmaX * m_sigmaX);                                            
	float x = (-1.f)*(float)(conv_size);                                                         
	float tsum = 0.0f;     
	if(m_wx)
		delete [] m_wx;
	m_wx = new float[nn];

   //generate gaussian mask coefficients                                
	for(i=0; i < nn; i++) 
	{                                                                         
		m_wx[i] = exp(s2 * x * x);                                            
		tsum += m_wx[i];                                                           
		x += 1.0f;                                                               
	} 

	for (i=0; i < nn; i++) 
		m_wx[i] = m_wx[i]/tsum; 

	m_conv_sizeX = conv_size;


	conv_size = (int) (m_sigmaY*3.0f); // mask is 3 dSigma on each side      
	nn = 2 * conv_size +1;                                                  
	s2 = -0.5f / (m_sigmaY * m_sigmaY);                                            
	x = (-1.f)*(float)(conv_size);                                                         
	tsum = 0.0f; 
	if(m_wy)
		delete [] m_wy;
	m_wy = new float[nn];

   //generate gaussian mask coefficients                                
	for(i=0; i < nn; i++) 
	{                                                                         
		m_wy[i] = exp(s2 * x * x);                                            
		tsum += m_wy[i];                                                           
		x += 1.0f;                                                               
	} 

	for (i=0; i < nn; i++) 
		m_wy[i] = m_wy[i]/tsum; 

	m_conv_sizeY = conv_size;

	if(m_sigmaZ == 0.0f)
		return;

	conv_size = (int) (m_sigmaZ*3.0f); // mask is 3 dSigma on each side      
	nn = 2 * conv_size +1;                                                  
	s2 = -0.5f / (m_sigmaZ * m_sigmaZ);                                            
	x = (-1.f)*(float)(conv_size);                                                         
	tsum = 0.0f; 
	if(m_wz)
		delete [] m_wz;
	m_wz = new float[nn];

   //generate gaussian mask coefficients                                
	for(i=0; i < nn; i++) 
	{                                                                         
		m_wz[i] = exp(s2 * x * x);                                            
		tsum += m_wz[i];                                                           
		x += 1.0f;                                                               
	} 

	for (i=0; i < nn; i++) 
		m_wz[i] = m_wz[i]/tsum; 

	m_conv_sizeZ = conv_size;

}

void GaussianFilter::filter(float **plane, int nrows, int ncols)
{

	int j, k, nn;
	int jj, kk;
	int ind;
	int limit;
	float tsum;

	//int nslices = vol.slices();
	//int nrows = plane.rows();
	//int ncols = plane.cols();


	float **temp = new float*[nrows];
	temp[0] = new float[nrows*ncols];
	for(j = 1; j < nrows; j++)
	{
		temp[j] = temp[j-1] + ncols;
	}



	//blur along x dimension
	#pragma omp parallel for private(j, k, kk, tsum, limit, ind, nn) 
	for(j = 0; j < nrows; j++)
	{
		for(k = m_conv_sizeX; k < ncols - m_conv_sizeX; k++)
		{
			temp[j][k] = 0.0;
			ind = 0;

			for(kk = k - m_conv_sizeX; kk <= k + m_conv_sizeX; kk++)
			{
				float a = temp[j][k];
				float b = m_wx[ind];
				float c = plane[j][kk];
				temp[j][k] += m_wx[ind++]*plane[j][kk];
			}
		}

		nn = 2 * m_conv_sizeX + 1;

		//leading edge
		for(k = 0; k < m_conv_sizeX; k++)
		{
			temp[j][k] = 0.0;
			tsum = 0.0;
			ind = nn - (m_conv_sizeX+k+1);
			limit = k+m_conv_sizeX;
			if(limit > ncols-1)
				limit = ncols-1;
			for(kk = 0; kk <= limit; kk++)
			{
				temp[j][k] += m_wx[ind]*plane[j][kk];
				tsum += m_wx[ind++];
			}
			temp[j][k] /= tsum;
		}

		//trailing edge
		for(k = ncols-m_conv_sizeX; k < ncols; k++)
		{
			temp[j][k] = 0.0;
			tsum = 0.0;
			ind = 0;
			limit = k - m_conv_sizeX;
			if(limit < 0)
				limit = 0;
			for(kk = limit; kk < ncols; kk++)
			{
				temp[j][k] += m_wx[ind]*plane[j][kk];
				tsum += m_wx[ind++];
			}
			temp[j][k] /= tsum;
		}

	}

	//blur along y dimension
	#pragma omp parallel for private(j, k, kk, tsum, limit, ind, nn) 
	for(k = 0; k < ncols; k++)
	{

		for(j = m_conv_sizeY; j < nrows - m_conv_sizeY; j++)
		{
			plane[j][k] = 0.0;
			ind = 0;

			for(jj = j - m_conv_sizeY; jj <= j + m_conv_sizeY; jj++)
			{
				plane[j][k] += m_wy[ind++]*temp[jj][k];
			}
		}

		nn = 2 * m_conv_sizeY +1;

		//leading edge
		for(j = 0; j < m_conv_sizeY; j++)
		{
			plane[j][k] = 0.0;
			tsum = 0.0;
			ind = nn - (m_conv_sizeY+j+1);
			limit = j + m_conv_sizeY;
			if(limit > nrows-1)
				limit = nrows-1;
			for(jj = 0; jj <= limit; jj++)
			{
				plane[j][k] += m_wy[ind]*temp[jj][k];
				tsum += m_wy[ind++];
			}
			plane[j][k] /= tsum;
		}

		//trailing edge
		for(j = nrows - m_conv_sizeY; j < nrows; j++)
		{
			plane[j][k] = 0.0;
			tsum = 0.0;
			ind = 0;
			limit = j - m_conv_sizeY;
			if(limit < 0)
				limit = 0;
			for(jj = limit; jj < nrows; jj++)
			{
				plane[j][k] += m_wy[ind]*temp[jj][k];
				tsum += m_wy[ind++];
			}
			plane[j][k] /= tsum;
		}
	}

}



void GaussianFilter::filter(float ***vol, int nslices, int nrows, int ncols)
{
	int i, j, k, nn;
	int ii, jj, kk;
	int ind;
	int limit, limit1;
	float tsum;


	//allocate temp volume
	float ***temp = new float **[nslices];
	for(int i = 0; i < nslices; i++)
	{
		temp[i] = new float*[nrows];
		temp[i][0] = new float[nrows*ncols];
		for(int j = 1; j < nrows; j++)
			temp[i][j] = temp[i][j-1] + ncols;
	}

	//blur each slice
	#pragma omp parallel for private(i, j, k, kk, tsum, limit, ind, nn) 
 	for(i = 0; i < nslices; i++)
	{
		//blur along x dimension
		for(j = 0; j < nrows; j++)
		{
			for(k = m_conv_sizeX; k < ncols-m_conv_sizeX; k++)
			{
				temp[i][j][k] = 0.0;
				ind = 0;

				for(kk = k-m_conv_sizeX; kk <= k+m_conv_sizeX; kk++)
				{
					temp[i][j][k] += m_wx[ind++]*vol[i][j][kk];
				}
			}

			nn = 2 * m_conv_sizeX + 1;

			//leading edge
			for(k = 0; k < m_conv_sizeX; k++)
			{
				temp[i][j][k] = 0.0;
				tsum = 0.0;
				ind = nn - (m_conv_sizeX+k+1);
				limit = k+m_conv_sizeX;
				if(limit > ncols-1)
					limit = ncols-1;
				for(kk = 0; kk <= limit; kk++)
				{
					temp[i][j][k] += m_wx[ind]*vol[i][j][kk];
					tsum += m_wx[ind++];
				}
				temp[i][j][k] /= tsum;
			}

			//trailing edge
			for(k = ncols-m_conv_sizeX; k < ncols; k++)
			{
				temp[i][j][k] = 0.0;
				tsum = 0.0;
				ind = 0;
				limit = k - m_conv_sizeX;
				if(limit < 0)
					limit = 0;
				for(kk = limit; kk < ncols; kk++)
				{
					temp[i][j][k] += m_wx[ind]*vol[i][j][kk];
					tsum += m_wx[ind++];
				}
				temp[i][j][k] /= tsum;
			}

		}

		//blur along y dimension
		for(k = 0; k < ncols; k++)
		{

			for(j = m_conv_sizeY; j < nrows-m_conv_sizeY; j++)
			{
				vol[i][j][k] = 0.0;
				ind = 0;

				for(jj = j-m_conv_sizeY; jj <= j+m_conv_sizeY; jj++)
				{
					vol[i][j][k] += m_wy[ind++]*temp[i][jj][k];
				}
			}

			nn = 2 * m_conv_sizeY + 1;

			//leading edge
			for(j = 0; j < m_conv_sizeY; j++)
			{
				vol[i][j][k] = 0.0;
				tsum = 0.0;
				ind = nn - (m_conv_sizeY+j+1);
				limit = j+m_conv_sizeY;
				if(limit > nrows-1)
					limit = nrows-1;
				for(jj = 0; jj <= limit; jj++)
				{
					vol[i][j][k] += m_wy[ind]*temp[i][jj][k];
					tsum += m_wy[ind++];
				}
				vol[i][j][k] /= tsum;
			}

			//trailing edge
			for(j = nrows-m_conv_sizeY; j < nrows; j++)
			{
				vol[i][j][k] = 0.0;
				tsum = 0.0;
				ind = 0;
				limit = j-m_conv_sizeY;
				if(limit < 0)
					limit = 0;
				for(jj = limit; jj < nrows; jj++)
				{
					vol[i][j][k] += m_wy[ind]*temp[i][jj][k];
					tsum += m_wy[ind++];
				}
				vol[i][j][k] /= tsum;
			}
		}
	}


	if(m_conv_sizeZ == 0)
		return;

	for(i = 0; i < nslices; i++)
	{
		for(j = 0; j < nrows; j++)
		{
			for(k = 0; k < ncols; k++)
			{
				temp[i][j][k] = vol[i][j][k];
			}
		}
	}

	//blur in the z dimension
	#pragma omp parallel for private(i, j, k, ii, tsum, limit, limit1, ind, nn) 
	for(j = 0; j < nrows; j++)
	{
		for(k = 0; k < ncols; k++)
		{

			for(i = m_conv_sizeZ; i < nslices-m_conv_sizeZ; i++)
			{
				vol[i][j][k] = 0.0;
				ind = 0;

				for(ii = i-m_conv_sizeZ; ii <= i+m_conv_sizeZ; ii++)
				{
					vol[i][j][k] += m_wz[ind++]*temp[ii][j][k];
				}
			}

			nn = 2 * m_conv_sizeZ + 1;

			//leading edge
			if(m_conv_sizeZ > nslices-1)
				limit1 = nslices-1;
			else
				limit1 = m_conv_sizeZ;
			for(i = 0; i < limit1; i++)
			{
				vol[i][j][k] = 0.0;
				tsum = 0.0;
				ind = nn - (m_conv_sizeZ+i+1);
				limit = i+m_conv_sizeZ;
				if(limit > nslices-1)
					limit = nslices-1;
				for(ii = 0; ii <= limit; ii++)
				{
					vol[i][j][k] += m_wz[ind]*temp[ii][j][k];
					tsum += m_wz[ind++];
				}
				vol[i][j][k] /= tsum;
			}

			//trailing edge
			limit1 = nslices-m_conv_sizeZ;
			if(limit1 < 0)
				limit1 = 0;
			for(i = limit1; i < nslices; i++)
			{
				vol[i][j][k] = 0.0;
				tsum = 0.0;
				ind = 0;
				limit = i-m_conv_sizeZ;
				if(limit < 0)
					limit = 0;
				for(ii = limit; ii < nslices; ii++)
				{
					vol[i][j][k] += m_wz[ind]*temp[ii][j][k];
					tsum += m_wz[ind++];
				}
				vol[i][j][k] /= tsum;
			}
		}
	}

	//deallocate temp volume
	for(int i = 0; i < nslices; i++)
	{
		delete [] temp[i][0];
		delete [] temp[i];
	}

	delete [] temp;

}

//void GaussianFilter::filter(float ***vol, int nslices, int nrows, int ncols)
//{
//	int i, j, k, nn;
//	int ii, jj, kk;
//	int ind;
//	int limit, limit1;
//	float tsum;
//
//
//	//allocate temp volume
//	//float ***temp = new float **[nslices];
//	//for (int i = 0; i < nslices; i++)
//	//{
//	//	temp[i] = new float*[nrows];
//	//	temp[i][0] = new float[nrows*ncols];
//	//	for (int j = 1; j < nrows; j++)
//	//		temp[i][j] = temp[i][j - 1] + ncols;
//	//}
//	
//	float **slice = new float*[nrows];
//	slice[0] = new float[nrows*ncols];
//	for (int i = 1; i < nrows; i++)
//	{
//		slice[i] = slice[i - 1] + ncols;
//	}
//
//	printf("1\n");
//	//blur each slice
////#pragma omp parallel for private(i, j, k, kk, tsum, limit, ind, nn) 
//	for (i = 0; i < nslices; i++)
//	{
//		//blur along x dimension
//		#pragma omp parallel for private(j, k, kk, tsum, limit, ind, nn) 
//		for (j = 0; j < nrows; j++)
//		{
//			for (k = m_conv_sizeX; k < ncols - m_conv_sizeX; k++)
//			{
//				//temp[i][j][k] = 0.0;
//				slice[j][k] = 0.0f;
//				ind = 0;
//
//				for (kk = k - m_conv_sizeX; kk <= k + m_conv_sizeX; kk++)
//				{
//					//temp[i][j][k] += m_wx[ind++] * vol[i][j][kk];
//					slice[j][k] += m_wx[ind++] * vol[i][j][kk];
//				}
//			}
//
//			nn = 2 * m_conv_sizeX + 1;
//
//			//leading edge
//			for (k = 0; k < m_conv_sizeX; k++)
//			{
//				//temp[i][j][k] = 0.0;
//				slice[j][k] = 0.0;
//				tsum = 0.0;
//				ind = nn - (m_conv_sizeX + k + 1);
//				limit = k + m_conv_sizeX;
//				if (limit > ncols - 1)
//					limit = ncols - 1;
//				for (kk = 0; kk <= limit; kk++)
//				{
//					//temp[i][j][k] += m_wx[ind] * vol[i][j][kk];
//					slice[j][k] += m_wx[ind] * vol[i][j][kk];
//					tsum += m_wx[ind++];
//				}
//				//temp[i][j][k] /= tsum;
//				slice[j][k] /= tsum;
//			}
//
//			//trailing edge
//			for (k = ncols - m_conv_sizeX; k < ncols; k++)
//			{
//				//temp[i][j][k] = 0.0;
//				slice[j][k] = 0.0;
//				tsum = 0.0;
//				ind = 0;
//				limit = k - m_conv_sizeX;
//				if (limit < 0)
//					limit = 0;
//				for (kk = limit; kk < ncols; kk++)
//				{
//					//temp[i][j][k] += m_wx[ind] * vol[i][j][kk];
//					slice[j][k] += m_wx[ind] * vol[i][j][kk];
//					tsum += m_wx[ind++];
//				}
//				//temp[i][j][k] /= tsum;
//				slice[j][k] /= tsum;
//			}
//
//		}
//
//		//blur along y dimension
//		for (k = 0; k < ncols; k++)
//		{
//
//			for (j = m_conv_sizeY; j < nrows - m_conv_sizeY; j++)
//			{
//				vol[i][j][k] = 0.0;
//				ind = 0;
//
//				for (jj = j - m_conv_sizeY; jj <= j + m_conv_sizeY; jj++)
//				{
//					//vol[i][j][k] += m_wy[ind++] * temp[i][jj][k];
//					vol[i][j][k] += m_wy[ind++] * slice[jj][k];
//				}
//			}
//
//			nn = 2 * m_conv_sizeY + 1;
//
//			//leading edge
//			for (j = 0; j < m_conv_sizeY; j++)
//			{
//				vol[i][j][k] = 0.0;
//				tsum = 0.0;
//				ind = nn - (m_conv_sizeY + j + 1);
//				limit = j + m_conv_sizeY;
//				if (limit > nrows - 1)
//					limit = nrows - 1;
//				for (jj = 0; jj <= limit; jj++)
//				{
//					//vol[i][j][k] += m_wy[ind] * temp[i][jj][k];
//					vol[i][j][k] += m_wy[ind] * slice[jj][k];
//					tsum += m_wy[ind++];
//				}
//				vol[i][j][k] /= tsum;
//			}
//
//			//trailing edge
//			for (j = nrows - m_conv_sizeY; j < nrows; j++)
//			{
//				vol[i][j][k] = 0.0;
//				tsum = 0.0;
//				ind = 0;
//				limit = j - m_conv_sizeY;
//				if (limit < 0)
//					limit = 0;
//				for (jj = limit; jj < nrows; jj++)
//				{
//					//vol[i][j][k] += m_wy[ind] * temp[i][jj][k];
//					vol[i][j][k] += m_wy[ind] * slice[jj][k];
//					tsum += m_wy[ind++];
//				}
//				vol[i][j][k] /= tsum;
//			}
//		}
//	}
//
//	delete[] slice[0];
//	delete[] slice;
//
//	if (m_conv_sizeZ == 0)
//		return;
//
//	//for (i = 0; i < nslices; i++)
//	//{
//	//	for (j = 0; j < nrows; j++)
//	//	{
//	//		for (k = 0; k < ncols; k++)
//	//		{
//	//			temp[i][j][k] = vol[i][j][k];
//	//		}
//	//	}
//	//}
//
//	float *arr = new float[nslices];
//	printf("2\n");
//	//blur in the z dimension
////#pragma omp parallel for private(i, j, k, ii, tsum, limit, limit1, ind, nn) 
//	for (j = 0; j < nrows; j++)
//	{
//		for (k = 0; k < ncols; k++)
//		{
//
//			for (i = 0; i < nslices; i++)
//				arr[i] = vol[i][j][k];
//
//			for (i = m_conv_sizeZ; i < nslices - m_conv_sizeZ; i++)
//			{
//				vol[i][j][k] = 0.0;
//				ind = 0;
//
//				for (ii = i - m_conv_sizeZ; ii <= i + m_conv_sizeZ; ii++)
//				{
//					//vol[i][j][k] += m_wz[ind++] * temp[ii][j][k];
//					vol[i][j][k] += m_wz[ind++] * arr[ii];
//				}
//			}
//
//			nn = 2 * m_conv_sizeZ + 1;
//
//			//leading edge
//			if (m_conv_sizeZ > nslices - 1)
//				limit1 = nslices - 1;
//			else
//				limit1 = m_conv_sizeZ;
//			for (i = 0; i < limit1; i++)
//			{
//				vol[i][j][k] = 0.0;
//				tsum = 0.0;
//				ind = nn - (m_conv_sizeZ + i + 1);
//				limit = i + m_conv_sizeZ;
//				if (limit > nslices - 1)
//					limit = nslices - 1;
//				for (ii = 0; ii <= limit; ii++)
//				{
//					//vol[i][j][k] += m_wz[ind] * temp[ii][j][k];
//					vol[i][j][k] += m_wz[ind] * arr[ii];
//					tsum += m_wz[ind++];
//				}
//				vol[i][j][k] /= tsum;
//			}
//
//			//trailing edge
//			limit1 = nslices - m_conv_sizeZ;
//			if (limit1 < 0)
//				limit1 = 0;
//			for (i = limit1; i < nslices; i++)
//			{
//				vol[i][j][k] = 0.0;
//				tsum = 0.0;
//				ind = 0;
//				limit = i - m_conv_sizeZ;
//				if (limit < 0)
//					limit = 0;
//				for (ii = limit; ii < nslices; ii++)
//				{
//					//vol[i][j][k] += m_wz[ind] * temp[ii][j][k];
//					vol[i][j][k] += m_wz[ind] * arr[ii];
//					tsum += m_wz[ind++];
//				}
//				vol[i][j][k] /= tsum;
//			}
//		}
//	}
//
//	delete[] arr;
//
//}

