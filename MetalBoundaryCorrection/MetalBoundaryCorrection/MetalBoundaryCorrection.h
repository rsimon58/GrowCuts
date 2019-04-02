#pragma once
//using namespace std;
#include <vector_types.h> // Required to include CUDA vector types

class MetalBoundaryCorrection
{

protected:

	float **m_mask;
	float **m_image; 
	float **m_newMask;
	int **m_labels, **m_newLabels;
	float **m_strength, **m_newStrength;
	int m_ncols, m_nrows;
	int m_left, m_right, m_top, m_bottom;
	int m_nDilate;
	bool m_bUsingGPU;

	//cuda variables
	dim3 m_GridSize, m_BlockSize;
	int *d_labels, *d_newLabels;
	float *d_strength, *d_newStrength;
	size_t d_labelPitch, d_strengthPitch;
	float *d_image;
	//float *d_mask;
	size_t d_imagePitch;
	int *d_converged;

public:

	MetalBoundaryCorrection();
	~MetalBoundaryCorrection();

	void setup(int nrows, int ncols, bool bUsingGPU);
	void FindBoundary(float **image, float **mask);

protected:

	void initialize();
	void dealloc();
	void setInitalLabelsAndStrength();
	void growcuts();
	void dilateMask(int nDilate);
	void initializeCPU();

	void growcutsGPU();
	void growcutsGPU_Checker();
	void initializeGPU();
	void cleanUpCuda();

};

