#pragma once



class  GaussianFilter
{

public:
	GaussianFilter();
	GaussianFilter(float sigmaX, float sigmaY, float sigmaZ = 0.0f); 
	~GaussianFilter();

	void setSigmaX(float sigma) { m_sigmaX = sigma;};
	void setSigmaY(float sigma) { m_sigmaY = sigma;};
	void setSigmaZ(float sigma) { m_sigmaZ = sigma;};

	void initializeFilter();
	void initializeFilter(float sigmaX, float sigmaY, float sigmaZ = 0.0f);
	void filter(float **plane, int nrows, int ncols);
	//void filter(tplane<short> &plane);
	void filter(float ***vol, int nslices, int nrows, int ncols);

private:

	float m_sigmaX, m_sigmaY, m_sigmaZ;
	float *m_wx, *m_wy, *m_wz;
	int m_conv_sizeX, m_conv_sizeY, m_conv_sizeZ;

	void initialize();
	void dealloc();

};

