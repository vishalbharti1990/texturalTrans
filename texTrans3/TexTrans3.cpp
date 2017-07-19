// TexTrans3.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
#include<ctime>
#include<cmath>
#include<conio.h>           // may have to modify this line if not using Windows


template <typename T>
T **AllocateDynamicArray(int nRows, int nCols)
{
	T **dynamicArray;

	dynamicArray = new T*[nRows];
	for (int i = 0; i < nRows; i++)
		dynamicArray[i] = new T[nCols];

	return dynamicArray;
}

template <typename T>
void FreeDynamicArray(T** dArray)
{
	delete[] * dArray;
	delete[] dArray;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
 
	if (argc != 3){
		std::cout << "USAGE : TexTrans3 <input_image> <output_enhanced_image>";
		return 1;
	}

	cv::Mat imgOriginal;        // input image

	int start_s = clock();

	imgOriginal = cv::imread(argv[1], 0);          // open image, 0 for grayscale

	if (imgOriginal.empty()) {                                  // if unable to open image
		std::cout << "error: image not read from file\n\n";     // show error message on command line
		_getch();                                               // may have to modify this line if not using Windows
		return(0);                                              // and exit program
	}

	int **coMatN1 = AllocateDynamicArray<int>(256, 256);
	int **coMatN2 = AllocateDynamicArray<int>(256, 256);
	int **coMatN3 = AllocateDynamicArray<int>(256, 256);
	int **coMatN4 = AllocateDynamicArray<int>(256, 256);
	int pMat[256];

	for (int i = 0; i < 256; i++){
		pMat[i] = 1;
		for (int j = 0; j < 256; j++){
			coMatN1[i][j] = 1;
			coMatN2[i][j] = 1;
			coMatN3[i][j] = 1;
			coMatN4[i][j] = 1;
		}
	}

	int rows = imgOriginal.rows;
	int cols = imgOriginal.cols;
	
	try{
		for (int r = 0; r < rows; r++){
			for (int c = 0; c < cols; c++){
				pMat[(int)imgOriginal.at<uchar>(r, c)]++;
				if ((r + 1) < rows){
					coMatN4[(int)imgOriginal.at<uchar>(r, c)][(int)imgOriginal.at<uchar>(r + 1, c)]++;						//1
					if ((c + 1) < cols){
						coMatN1[(int)imgOriginal.at<uchar>(r, c)][(int)imgOriginal.at<uchar>(r + 1, c + 1)]++;				//2
					}
					if ((c - 1) >= 0){
						coMatN3[(int)imgOriginal.at<uchar>(r + 1, c - 1)][(int)imgOriginal.at<uchar>(r, c)]++;				//3
					}
				}
				if (r - 1 >= 0){
					coMatN4[(int)imgOriginal.at<uchar>(r - 1, c)][(int)imgOriginal.at<uchar>(r, c)]++;						//4
					if ((c + 1) < cols){
						coMatN3[(int)imgOriginal.at<uchar>(r, c)][(int)imgOriginal.at<uchar>(r - 1, c + 1)]++;				//5
					}
					if ((c - 1) >= 0){
						coMatN1[(int)imgOriginal.at<uchar>(r - 1, c - 1)][(int)imgOriginal.at<uchar>(r, c)]++;				//6
					}
				}
				if ((c + 1) < cols){
					coMatN2[(int)imgOriginal.at<uchar>(r, c)][(int)imgOriginal.at<uchar>(r, c + 1)]++;						//7
				}
				if ((c - 1) >= 0){
					coMatN2[(int)imgOriginal.at<uchar>(r, c - 1)][(int)imgOriginal.at<uchar>(r, c)]++;						//8
				}
			}
		}
	}catch (cv::Exception & e)
	{
		std::cerr << e.msg << std::endl; // output exception message
		return(1);
	}

	long n1Sum = 0;
	long n2Sum = 0;
	long n3Sum = 0;
	long n4Sum = 0;
	long pSum = 0;

	for (int i = 0; i < 256; i++){
		pSum += pMat[i];
		for (int j = 0; j < 256; j++){
			n1Sum += coMatN1[i][j];
			n2Sum += coMatN2[i][j];
			n3Sum += coMatN3[i][j];
			n4Sum += coMatN4[i][j];
		}
	}

	float **N1LogP = AllocateDynamicArray<float>(256, 256);
	float **N2LogP = AllocateDynamicArray<float>(256, 256);
	float **N3LogP = AllocateDynamicArray<float>(256, 256);
	float **N4LogP = AllocateDynamicArray<float>(256, 256);
	float logP[256];

	for (int i = 0; i < 256; i++){
		logP[i] = log((float)pMat[i] / pSum);
		for (int j = 0; j < 256; j++){
			N1LogP[i][j] = log((float)coMatN1[i][j] / n1Sum);
			N2LogP[i][j] = log((float)coMatN2[i][j] / n2Sum);
			N3LogP[i][j] = log((float)coMatN3[i][j] / n3Sum);
			N4LogP[i][j] = log((float)coMatN4[i][j] / n4Sum);
		}
	}

	cv::Mat outImage(rows, cols, CV_8UC1, cv::Scalar(255));

	float n1p, n2p, n3p, n4p;
	float **nPArr = AllocateDynamicArray<float>(rows, cols);

	try{
		for (int r = 0; r < rows; r++){
			for (int c = 0; c < cols; c++){
				n1p = n2p = n3p = n4p = 0;
				if ((r + 1) < rows){
					n4p += N4LogP[(int)imgOriginal.at<uchar>(r, c)][(int)imgOriginal.at<uchar>(r + 1, c)];								//1
					if ((c + 1) < cols){
						n1p += N1LogP[(int)imgOriginal.at<uchar>(r, c)][(int)imgOriginal.at<uchar>(r + 1, c + 1)];						//2
					}
					if ((c - 1) >= 0){
						n3p += N3LogP[(int)imgOriginal.at<uchar>(r + 1, c - 1)][(int)imgOriginal.at<uchar>(r, c)];						//3
					}
				}
				if (r - 1 >= 0){
					n4p += N4LogP[(int)imgOriginal.at<uchar>(r - 1, c)][(int)imgOriginal.at<uchar>(r, c)];								//4
					if ((c + 1) < cols){
						n3p += N3LogP[(int)imgOriginal.at<uchar>(r, c)][(int)imgOriginal.at<uchar>(r - 1, c + 1)];						//5
					}
					if ((c - 1) >= 0){
						n1p += N1LogP[(int)imgOriginal.at<uchar>(r - 1, c - 1)][(int)imgOriginal.at<uchar>(r, c)];						//6
					}
				}
				if ((c + 1) < cols)
					n2p += N2LogP[(int)imgOriginal.at<uchar>(r, c)][(int)imgOriginal.at<uchar>(r, c + 1)];								//7	
				if ((c - 1) >= 0)
					n2p += N2LogP[(int)imgOriginal.at<uchar>(r, c - 1)][(int)imgOriginal.at<uchar>(r, c)];								//8
				nPArr[r][c] = logP[(int)imgOriginal.at<uchar>(r, c)] + n1p + n2p + n3p + n4p;
			}
		}
	}catch (cv::Exception & e)
	{
		std::cerr << e.msg << std::endl; // output exception message
		return(1);
	}

	float min, max;

	min = max = nPArr[0][0];

	for (int r = 0; r < rows; r++){
		for (int c = 0; c < cols; c++){
			if (nPArr[r][c] < min){
				min = nPArr[r][c];
			}
			else if (nPArr[r][c] > max){
				max = nPArr[r][c];
			}
		}
	}

	for (int r = 0; r < rows; r++){
		for (int c = 0; c < cols; c++){
			outImage.at<uchar>(r, c) = std::round(((nPArr[r][c] - min) * 255) / (max - min));
		}
	}

	int stop_s = clock();
	std::cout << "time: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << std::endl;

	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);// Create a window for display.
	cv::imshow("Display window", outImage);                   // Show our image inside it.

	cv::waitKey(0);

	try{
		cv::imwrite(argv[2], outImage);
	}
	catch (cv::Exception & e)
	{
		std::cerr << e.msg << std::endl; // output exception message
	}

	std::cout << "\nEnhanced image " << argv[2] << " saved";

	return(0);
}





