#pragma once
#include <string>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <bitset>

using namespace std;

template <typename T> static void WriteImageSlice(const char* file, T*** proj, int numCol, int numRow, int slice);
template <typename T> static T*** AllocateVolume(int numCol, int numRow, int numFrame);
template <typename T> static T** AllocateImage(int ncols, int nrows);

template <typename T> static void FreeImage(T** image);

template <typename T> static void FreeVolume(T*** Image, int numFrame);
template <typename T> static void WriteImage(std::string dirIn, std::string fileIn, T*** proj, int numCol, int numRow, int numFrame);
template <typename T> static void ReadImage(std::string dirIn, std::string fileIn, T*** proj, int numCol, int numRow, int numFrame, int cropCols, int cropRows);
template <typename T> static void ReadImage(std::string dirIn, std::string fileIn, T*** proj, int numCol, int numRow, int numFrame, int cropColsLeft, int cropColsRight, int cropRowsTop, int cropRowsBottom);
template <typename T> void ReadImage(std::string dirIn, std::string *fileIn, float*** proj, int numCol, int numRow, int numFrame, int cropColsLeft, int cropColsRight, int cropRowsTop, int cropRowsBottom, int numSources, int sourceNumber, int intialProj);
template <typename T> void ReadImage2(std::string dirIn, std::string *fileIn, float*** proj, int numCol, int numRow, int numFrame, int cropColsLeft, int cropColsRight, int cropRowsTop, int cropRowsBottom, int numSources, int sourceNumber, int intialProj);
template <typename T> void ReadImageThreeSource(std::string dirIn, std::string fileIn[3], T*** proj, int numCol, int numRow, int numFrame, int cropCols, int cropRows);
template <typename T> void ReadImageThreeSource(std::string dirIn, std::string fileIn[3], T*** proj, int numCol, int numRow, int numFrame, int cropColsLeft, int cropColsRight, int cropRowsTop, int cropRowsBottom);
template <typename T> static void ReadBitImage(std::string dirIn, std::string fileIn, T*** image, int numCol, int numRow, int numFrame, int cropCols, int cropRows);
template <typename T> static void ReadImageLinear(std::string dirIn, std::string fileIn, T* image, int size);
template <typename T> static void WriteImageLinear(std::string dirIn, std::string fileIn, T* image, int size);
template <typename T> static void WriteImageLinearAppend(std::string dirIn, std::string fileIn, T* image, int size);
template <typename T> static void WriteImageCoronal(std::string dirIn, std::string fileIn, T*** proj, int nx, int ny, int nz);
template <typename T> static void WriteImageSagital(std::string dirIn, std::string fileIn, T*** proj, int nx, int ny, int nz);

template <class T> T ***Create3D(int N3, int N2, int N1);
template <class T> void Delete3D(T ***array);

static void NormalizeByImage(float*** proj, float*** airscan, int nx, int ny, int nz);
static void NormalizeByConstant(float*** proj, double *i0, int numCol, int numRow, int numFrame);
static void LogProjections(float*** proj, int numCol, int numRow, int numFrame);
template <typename T> T*** AllocateVolume(int numCol, int numRow, int numFrame)
{
    try{
        T*** Image = new T**[numFrame];
        for (int i = 0; i< numFrame; i++)
        {
            Image[i] = new T*[numRow];
			Image[i][0] = new T[numRow*numCol];
			for(int j = 1; j < numRow; j++)
				Image[i][j] = Image[i][j-1] + numCol;
        }
       return Image;
    }
    catch(std::bad_alloc)
    {
        std::cerr<<"Not enough memory could be allocated by the system."<<std::endl;    //TODO: other recovery strategies??
        assert(false);
        return NULL;
    }
}
template <typename T> T** AllocateImage(int ncols, int nrows)
{
	try
	{
		T** ptr = new T*[nrows];
		T* pool = new T[nrows*ncols];
		for (unsigned i = 0; i < nrows; ++i, pool += ncols )
			ptr[i] = pool;
		return ptr;
	}
	catch(std::bad_alloc)
	{
		std::cerr<<"Not enough memory could be allocated by the system."<<std::endl;    //TODO: other recovery strategies??
		assert(false);
		return NULL;
	}
}
template <typename T> void FreeVolume(T*** Image, int numFrame)
{
    if(Image == NULL)
        return;
    for (int i = 0; i < numFrame; i++) 
        {
            if(Image[i] != NULL)
            {
                delete [] Image[i][0];
				delete [] Image[i];
                Image[i]= NULL;
            }
    }
    delete[] Image;
    Image= NULL;
}
template <typename T> void FreeImage(T** image)
{
    if(image == NULL)
        return;

    if(image != NULL)
    {
        delete [] image[0];
		delete [] image;
        image= NULL;
    }
    delete[] image;
    image= NULL;
}
template <typename T> void WriteImageSlice(const char* file, T*** proj, int numCol, int numRow, int slice)
{
	FILE *fp;
	fopen_s(&fp,file, "wb");
	fwrite(&proj[slice][0][0], numRow*numCol*sizeof(T), 1, fp);
	fclose(fp);
}
template <typename T> void WriteImage(std::string dirIn, std::string fileIn, T*** proj, int numCol, int numRow, int numFrame)
{
	string file = dirIn + fileIn;	
	FILE *fp;
	fopen_s(&fp,file.c_str(), "wb"); // ab

	for(int i = 0; i < numFrame; i++)
	{
		fwrite(&proj[i][0][0], numRow*numCol*sizeof(T), 1, fp);
	}

	fclose(fp);
}
template <typename T> void WriteImageSagital(std::string dirIn, std::string fileIn, T*** proj, int nx, int ny, int nz)
{
	string file = dirIn + fileIn;
	FILE *fp;
	fopen_s(&fp,file.c_str(), "wb"); // ab

	for(int x = nx - 1; x >= 0; x--)
		for(int z = 0; z < nz; z++)
			for(int y = 0; y < ny; y++)
				fwrite(&proj[z][y][x], sizeof(T), 1, fp);


	fclose(fp);
}
template <typename T> void ReadImage(std::string dirIn, std::string fileIn, float*** proj, int numCol, int numRow, int numFrame, int cropCols, int cropRows)
{
	string file = dirIn + fileIn;
	FILE *fp;
	fopen_s(&fp,file.c_str(), "rb");
	T *temp = new T[numRow*numCol];
	for(int i = 0; i < numFrame; i++)
	{
		int num = fread (temp, sizeof(T),numRow*numCol,fp);
		for(int j = cropRows; j < numRow-cropRows; j++)
		{
			for(int k = cropCols; k < numCol-cropCols; k++)
			{
				proj[i][j-cropRows][k-cropCols] = (float)temp[j*numCol + k];
	
			}
		}
	}
	free(temp);
	fclose(fp);
}
//template <typename T> void ReadImageThreeSource(std::string dirIn, std::string fileIn[3], float*** proj, int numCol, int numRow, int numFrame, int cropCols, int cropRows)
//{
//	string file[3];
//	file[0] = dirIn + fileIn[0];
//	file[1] = dirIn + fileIn[1];
//	file[2] = dirIn + fileIn[2];
//	
//	FILE *fp[3];
//	fopen_s(&fp[0],file.c_str(), "rb");
//	fopen_s(&fp[1],file.c_str(), "rb");
//	fopen_s(&fp[2],file.c_str(), "rb");
//
//	int iSource = 0;
//	T *temp = new T[numRow*numCol];
//	for(int i = 0; i < numFrame; i++)
//	{
//
//		int num = fread (temp, sizeof(T),numRow*numCol,fp[iSource]);
//		for(int j = cropRows; j < numRow-cropRows; j++)
//		{
//			for(int k = cropCols; k < numCol-cropCols; k++)
//			{
//				proj[i][j-cropRows][k-cropCols] = (float)temp[j*numCol + k];
//	
//			}
//		}
//		iSource++;
//		if(iSource > 2)
//			iSource = 0;
//	}
//	free(temp);
//	fclose(fp);
//}
template <typename T> void ReadImage(std::string dirIn, std::string fileIn, float*** proj, int numCol, int numRow, int numFrame, int cropColsLeft, int cropColsRight, int cropRowsTop, int cropRowsBottom)
{
	string file = dirIn + fileIn;
	FILE *fp;
	fopen_s(&fp,file.c_str(), "rb");
	T *temp = new T[numRow*numCol];
	for(int i = 0; i < numFrame; i++)
	{
		int num = fread (temp, sizeof(T),numRow*numCol,fp);
		for(int j = cropRowsTop; j < numRow-cropRowsBottom; j++)
		{
			for(int k = cropColsLeft; k < numCol-cropColsRight; k++)
			{
				proj[i][j-cropRowsTop][k-cropColsLeft] = (float)temp[j*numCol + k];
	
			}
		}
	}
	free(temp);
	fclose(fp);
}
template <typename T> void ReadImage(std::string dirIn, std::string *fileIn, float*** proj, int numCol, int numRow, int numFrame, int cropColsLeft, int cropColsRight, int cropRowsTop, int cropRowsBottom, int numSources, int sourceNumber, int intialProj)
{
	//string file[numSources];
	string *file = new string[numSources];

	//FILE *fp[numSources];
	FILE **fp = new FILE*[numSources];

	T *temp = new T[numRow*numCol];

	int offset = sizeof(T)*numRow*numCol*intialProj;
	for(int i = 0; i < numSources; i++)
	{
		if(numSources == 1)
			file[i] = dirIn + fileIn[sourceNumber];
		else
			file[i] = dirIn + fileIn[i];
		fopen_s(&fp[i],file[i].c_str(), "rb");
		fseek( fp[i], offset, SEEK_SET );
	}

	int framesPerSource = numFrame / numSources;
	for(int i = 0; i < framesPerSource; i++)
	{
		for(int s = 0; s < numSources; s++)
		{
			int num = fread (temp, sizeof(T),numRow*numCol,fp[s]);

			for(int j = cropRowsTop; j < numRow-cropRowsBottom; j++)
			{
				for(int k = cropColsLeft; k < numCol-cropColsRight; k++)
				{
					proj[i * numSources + s][j-cropRowsTop][k-cropColsLeft] = (float)temp[j*numCol + k];
				}
			}
		}
	}
	free(temp);
	for(int s = 0; s < numSources; s++)
		fclose(fp[s]);
}
template <typename T> void ReadImageThreeSource(std::string dirIn, std::string fileIn[3], float*** proj, int numCol, int numRow, int numFrame, int cropColsLeft, int cropColsRight, int cropRowsTop, int cropRowsBottom)
{
	string file[3];
	file[0] = dirIn + fileIn[0];
	file[1] = dirIn + fileIn[1];
	file[2] = dirIn + fileIn[2];
	
	FILE *fp[3];
	fopen_s(&fp[0],file[0].c_str(), "rb");
	fopen_s(&fp[1],file[1].c_str(), "rb");
	fopen_s(&fp[2],file[2].c_str(), "rb");

	int iSource = 0;
	T *temp = new T[numRow*numCol];
	for(int i = 0; i < numFrame; i++)
	{

		int num = fread (temp, sizeof(T),numRow*numCol,fp[iSource]);
		for(int j = cropRowsTop; j < numRow-cropRowsBottom; j++)
		{
			for(int k = cropColsLeft; k < numCol-cropColsRight; k++)
			{
				proj[i][j-cropRowsTop][k-cropColsLeft] = (float)temp[j*numCol + k];
			}
		}
		iSource++;
		if(iSource > 2)
			iSource = 0;
	}
	free(temp);
	fclose(fp[0]);
	fclose(fp[1]);
	fclose(fp[2]);
}

template <typename T> void ReadBitImage(std::string dirIn, std::string fileIn, T*** image, int numCol, int numRow, int numFrame, int cropCols, int cropRows)
{
	string file = dirIn + fileIn;
	FILE *fp;
	fopen_s(&fp,file.c_str(), "rb");
	int size = (sizeof(T)*8);
	T *temp = new T[numRow*numCol/(sizeof(T)*8)];
	//T*** temp2 = AllocateImage<T>(numCol, numRow, numFrame);
	T*** temp2 = AllocateVolume<T>(numCol, numRow, numFrame);

	for(int i = 0; i < numFrame; i++)
	{
		int num = fread (temp, sizeof(T),numRow*numCol/(sizeof(T)*8),fp);
		for(int j = 0; j < numRow; j++)
		{
			for(int k = 0; k < (numCol)/(sizeof(T)*8); k++)
			{
				for (int b=0; b<sizeof(T)*8; b++)
				{
					temp2[i][j][(k)*sizeof(T)*8+b] = (temp[j*numCol/(sizeof(T)*8) + k] >> (7-b)) & 1;
				}
			}
		}
	}
	for(int i = 0; i < numFrame; i++)
	{
		for(int j = cropRows; j < numRow-cropRows; j++)
		{
			for(int k = cropCols; k < numCol-cropCols; k++)
			{
				image[i][j-cropRows][k-cropCols] = temp2[i][j][k];
	
			}
		}
	}

	free(temp);
	fclose(fp);
}
template <typename T> void ReadImageLinear(std::string dirIn, std::string fileIn, T* image, int size)
{
	string file = dirIn + fileIn;	
	FILE *fp;
	fopen_s(&fp,file.c_str(), "rb");
	fread (image, sizeof(T),size,fp);
	fclose(fp);
}
template <typename T> void WriteImageLinear(std::string dirIn, std::string fileIn, T* image, int size)
{
	string file = dirIn + fileIn;	
	FILE *fp;
	fopen_s(&fp,file.c_str(), "wb");
	fwrite (image, sizeof(T),size,fp);
	fclose(fp);
}
template <typename T> void WriteImageLinearAppend(std::string dirIn, std::string fileIn, T* image, int size)
{
	string file = dirIn + fileIn;
	FILE *fp;
	fopen_s(&fp,file.c_str(), "ab");
	fwrite (image, sizeof(T),size,fp);
	fclose(fp);
}
template <typename T> void WriteImageCoronal(std::string dirIn, std::string fileIn, T*** proj, int nx, int ny, int nz)
{
	string file = dirIn + fileIn;
	FILE *fp;
	fopen_s(&fp,file.c_str(), "wb"); // ab

	for(int y = 0; y < ny; y++)
		for(int z = 0; z < nz; z++)
			for(int x = 0; x < nx; x++)
				fwrite(&proj[z][y][x], sizeof(T), 1, fp);

	fclose(fp);
}

template <class T> T ***Create3D(int N3, int N2, int N1)
{
    T *** array = new T ** [N1];

    array[0] = new T * [N1*N2];

    array[0][0] = new T [N1*N2*N3];

    int i,j,k;

    for( i = 0; i < N1; i++) {

        if (i < N1 -1 ) {

            array[0][(i+1)*N2] = &(array[0][0][(i+1)*N3*N2]);

            array[i+1] = &(array[0][(i+1)*N2]);

        }

        for( j = 0; j < N2; j++) {     
            if (j > 0) array[i][j] = array[i][j-1] + N3;
        }

    }

    cout << endl;
    return array;
};

template <class T> void Delete3D(T ***array) 
{
    delete[] array[0][0]; 
    delete[] array[0];
    delete[] array;
};

void NormalizeByConstant(float*** proj, double *i0, int numCol, int numRow, int numFrame)
{
	float offset, ans;

	for(int i = 0; i < numFrame; i++)
	{
		offset = i0[i];

		for(int j = 0;  j < numRow; j++)
		{
			for(int k = 0; k < numCol; k++)
			{
				ans = proj[i][j][k] / offset;
				if(ans > 1.0)
					ans = 1.0;
				proj[i][j][k] = ans;
			}
		}
	}
}
void NormalizeByImage(float*** proj, float*** airscan, int nx, int ny, int nz)
{
	float ans;
	for(int y = 0; y < ny; y++)
		for(int z = 0; z < nz; z++)
			for(int x = 0; x < nx; x++)
			{
				ans = proj[z][y][x] / airscan[z][y][x];
				if(ans > 1.0)
					ans = 1.0;
				proj[z][y][x] = ans;
			}
}
void LogProjections(float*** proj, int numCol, int numRow, int numFrame)
{
	// log projections
	for(int p = 0; p < numFrame; p++)
	{
		for(int v = 0; v < numRow; v++)
		{
			for(int u = 0; u < numCol; u++)
			{
				proj[p][v][u] = -log(proj[p][v][u]);
				if(proj[p][v][u] < 0.00000001)
					proj[p][v][u] = 0.0;
			}
		}
	}
}

template <typename T> void ReadImage2(std::string dirIn, std::string *fileIn, float*** proj, int numCol, int numRow, int numFrame, int cropColsLeft, int cropColsRight, int cropRowsTop, int cropRowsBottom, int numSources, int sourceNumber, int intialProj)
{
	//string file[numSources];
	string *file = new string[numSources];

	//FILE *fp[numSources];
	FILE **fp = new FILE*[numSources];

	T *temp = new T[numRow*numCol];

	int offset = sizeof(T)*numRow*numCol*intialProj;
	for(int i = 0; i < numSources; i++)
	{
		if(numSources == 1)
			file[i] = dirIn + fileIn[sourceNumber];
		else
			file[i] = dirIn + fileIn[i];
		fopen_s(&fp[i],file[i].c_str(), "rb");
		fseek( fp[i], offset, SEEK_SET );
	}

	int framesPerSource = numFrame / numSources;
	for(int s = 0; s < numSources; s++)
	{

		for(int i = 0; i < framesPerSource; i++)
		{
			int num = fread (temp, sizeof(T),numRow*numCol,fp[s]);

			for(int j = cropRowsTop; j < numRow-cropRowsBottom; j++)
			{
				for(int k = cropColsLeft; k < numCol-cropColsRight; k++)
				{
					proj[i + s * framesPerSource][j-cropRowsTop][k-cropColsLeft] = (float)temp[j*numCol + k];
				}
			}
		}
	}
	free(temp);
	for(int s = 0; s < numSources; s++)
		fclose(fp[s]);
}