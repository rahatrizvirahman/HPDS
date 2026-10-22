#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

__global__ void BGRToGrayScale(uchar3 *BGRImage, unsigned char *grayImage, int numberRows, int numberColumns)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int    row = blockIdx.y * blockDim.y + threadIdx.y;

    if(column < numberColumns && row < numberRows)
    {
		uchar3 pixel = BGRImage[row * numberColumns + column];

		unsigned char grayValue = (0.114f * pixel.x) + (0.587f * pixel.y) + (0.299f * pixel.z); // BGR to gray

		grayImage[row * numberColumns + column] = grayValue;
    }
}

int main(int argc, char **argv)
{
	if(argc != 3) {
		cerr << "Usage: ./program input.jpg output.jpg" << endl;
		return 1;
	}

	string inputImageFileName = string(argv[1]);
	string outputImageFileName = string(argv[2]);

    // Read image
	cv::Mat inputImage = cv::imread(inputImageFileName, IMREAD_COLOR);
	cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);

    // Allocate GPU memory
    uchar3 *d_BGRImage;
    unsigned char *d_grayscaleImage;

    cudaMalloc(&d_BGRImage, sizeof(uchar3) * inputImage.rows * inputImage.cols);
    cudaMalloc(&d_grayscaleImage, sizeof(unsigned char) * inputImage.rows * inputImage.cols);

	cudaMemcpy(d_BGRImage, (uchar3*) inputImage.ptr<uchar3>(0), sizeof(uchar3) * inputImage.rows * inputImage.cols, cudaMemcpyHostToDevice);

	int threadsPerBlockPerDim = 16;
	int gridDimx = (inputImage.cols + threadsPerBlockPerDim - 1) / threadsPerBlockPerDim;
	int gridDimy = (inputImage.rows + threadsPerBlockPerDim - 1) / threadsPerBlockPerDim;

	dim3 blockDim(threadsPerBlockPerDim, threadsPerBlockPerDim);
	dim3 gridDim(gridDimx,gridDimy);

	BGRToGrayScale <<< gridDim, blockDim >>> (d_BGRImage, d_grayscaleImage, inputImage.rows, inputImage.cols);

	cudaMemcpy((unsigned char*) outputImage.ptr<unsigned char>(0), d_grayscaleImage, sizeof(unsigned char) * inputImage.rows * inputImage.cols, cudaMemcpyDeviceToHost);

    cv::imwrite(outputImageFileName, outputImage);

    // Free Memory
    cudaFree(d_BGRImage);
    cudaFree(d_grayscaleImage);

    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

    return 0;
}
