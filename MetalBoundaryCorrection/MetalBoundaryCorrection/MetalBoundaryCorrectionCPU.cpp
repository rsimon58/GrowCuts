#include <cstdlib>
#include <ctime>
#include <string>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include "gaussianFilter.h"
#include "ImageIO.h"
#include "MetalBoundaryCorrectionCPU.h"

//using namespace std;

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrectionCPU::initialize()
{
	m_labels = NULL;
	m_newLabels = NULL;
	m_strength = NULL;
	m_newStrength = NULL;

	m_nDilate = 0;
}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrectionCPU::dealloc()
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


}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
MetalBoundaryCorrectionCPU::MetalBoundaryCorrectionCPU()
{
	initialize();
}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
MetalBoundaryCorrectionCPU::~MetalBoundaryCorrectionCPU()
{
	dealloc();
}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrectionCPU::setup(int rows, int cols, bool bUsingGPU)
{
	m_ncols = cols;
	m_nrows = rows;


	initializeCPU();
}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrectionCPU::initializeCPU()
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
void MetalBoundaryCorrectionCPU::FindBoundary(float **image, float **mask)
{
	m_mask = mask;
	m_image = image;

	//test();

	setInitalLabelsAndStrength();

	//test();

	growcuts();


	if (m_nDilate > 0)
		dilateMask(m_nDilate);
}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrectionCPU::test()
{
	int i, j;
	float sum = 0;
	float sum2 = 0.0f;
	int cnt = 0;

	for (i = 0; i < m_nrows; i++)
	{
		for (j = 0; j < m_ncols; j++)
		{
			if (m_mask[i][j] > 0.5f)
			{
				sum += m_image[i][j];
				sum2 += m_image[i][j] * m_image[i][j];
				cnt++;
			}
		}
	}

	float avg = sum / (float)cnt;
	float sd = sum2 / (float)cnt - avg * avg;
	sd = sqrt(sd);

	printf("avg %f sd %f\n", avg, sd);

	float lo = avg - sd;
	float hi = avg + sd;

	for (i = 0; i < m_nrows; i++)
	{
		for (j = 0; j < m_ncols; j++)
		{
			if (m_mask[i][j] == 1.0f)
			{
				if ((m_image[i][j] > hi))//|| (m_image[i][j] < lo) )
				{
					m_strength[i][j] = 1.0;//0.9f;
					m_labels[i][j] = -1;  //flip label
				}
			}
			else
				m_mask[i][j] = 0.0f;
		}
	}

}

////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////
void MetalBoundaryCorrectionCPU::setInitalLabelsAndStrength()
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
	for (i = 1; i < m_nrows - 1; i++)
	{
		for (j = 1; j < m_ncols - 1; j++)
		{
			if (m_mask[i][j] == 0)
			{
				if (m_mask[i][j - 1] != 0 || m_mask[i][j + 1] != 0 || m_mask[i - 1][j] != 0 || m_mask[i + 1][j] != 0)
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
			else if (m_mask[i][j] == 1)
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
void MetalBoundaryCorrectionCPU::dilateMask(int nDilate)
{
	int i, j;

	for (int k = 0; k < nDilate; k++)
	{
		memcpy(m_strength[0], m_mask[0], sizeof(float)*m_nrows*m_ncols);

		for (i = m_top - 1; i < m_bottom + 1; i++)
		{
			for (j = m_left - 1; j < m_right + 1; j++)
			{
				if (m_mask[i][j] == 0)
				{
					if (m_mask[i][j - 1] != 0 || m_mask[i][j + 1] != 0 || m_mask[i - 1][j] != 0 || m_mask[i + 1][j] != 0)
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
void MetalBoundaryCorrectionCPU::growcuts()
{

	int i, j, m;
	int ii, jj;
	//int Nx[] = { -1, 1,  0, 0, -1, -1, 1,  1 }; //8-neighbors
	//int Ny[] = {  0, 0, -1, 1,  1, -1, 1, -1 };
	//int nNeighbors = 8;
	int Nx[] = { -1, 1,  0, 0 }; //4-neighbors
	int Ny[] = { 0, 0, -1, 1 };
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
		for (i = m_top - 1; i < m_bottom + 1; i++)
		{
			for (j = m_left - 1; j < m_right + 1; j++)
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

		//lets not go crazy...
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

	printf("total iterations=%d\n", its);
	printf("C %f\n", max);



}
