#pragma once
//using namespace std;

class MetalBoundaryCorrectionCPU
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



public:

	MetalBoundaryCorrectionCPU();
	~MetalBoundaryCorrectionCPU();

	void setup(int nrows, int ncols);
	void FindBoundary(float **image, float **mask);

protected:

	void initialize();
	void dealloc();
	void setInitalLabelsAndStrength();
	void growcuts();
	void test();
	void dilateMask(int nDilate);
	void initializeCPU();

};

#pragma once
