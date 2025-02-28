// MetalBoundaryCorrection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <time.h>
#include "ImageIO.h"
#include "MetalBoundaryCorrectionCPU.h"
#include "MetalBoundaryCorrection.h"

int main()
{

	int nrows = 884;
	int ncols = 1076;
	int nprojs = 10;

	string dir = "C:\\MetalBoundaryCorrection\\Data\\";
	string projFile = "image_UINT16_1076x884x10.raw";
	string maskFile = "mask0_UINT8_1076x884x10.raw";

	printf("read in Projection\n");
	float ***proj = AllocateVolume<float>(ncols, nrows, nprojs);
	ReadImage<unsigned short int>(dir, projFile, proj, ncols, nrows, nprojs, 0, 0);

	printf("read in Mask\n");
	float ***mask = AllocateVolume<float>(ncols, nrows, nprojs);
	ReadImage<unsigned char>(dir, maskFile, mask, ncols, nrows, nprojs, 0, 0);

	//normalize mask 0 - 1
	for (int p = 0; p < nprojs; p++)
	{
		for (int i = 0; i < nrows; i++)
		{
			for (int j = 0; j < ncols; j++)
			{
				mask[p][i][j] /= 255.0f;
			}
		}
	}

	MetalBoundaryCorrectionCPU myMBC;

	myMBC.setup(nrows, ncols);

	//MetalBoundaryCorrection myMBC;
	//bool bUsingGPU = true;
	//myMBC.setup(nrows, ncols, bUsingGPU);

	clock_t start0;
	start0 = clock();

	for (int i = 0; i < nprojs; i++)
	{
		printf("projection %d\n", i);
		myMBC.FindBoundary(proj[i], mask[i]);
	}

	maskFile = "MetalSegmentation_GrowCuts_" + to_string(ncols) + "x" + to_string(nrows) + "x" + to_string(nprojs) + ".raw";

    return 0;
}

