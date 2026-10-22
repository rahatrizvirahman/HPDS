#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define windowSize 3

__global__ void denoiseGPU(unsigned char *inputImage, unsigned char *outputImage, int numberRows, int numberColumns)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int    row = blockIdx.y * blockDim.y + threadIdx.y;

    if(column < numberColumns && row < numberRows)
    {
    	unsigned char window[windowSize * windowSize];

    	int effectivePixelsWithinWindow = 0;

    	for(int windowRow = 0; windowRow < windowSize; windowRow++) {
    		for(int windowColumn = 0; windowColumn < windowSize; windowColumn++) {
    			int pixelRow = row - (windowSize/2) + windowRow;
    			int pixelColumn = column - (windowSize/2) + windowColumn;

    			if(pixelRow >= 0 && pixelRow < numberRows && pixelColumn >= 0 && pixelColumn < numberColumns) {
    				window[effectivePixelsWithinWindow++] = inputImage[pixelRow * numberColumns + pixelColumn];
    			}
    			else {
    				// This pixel row/column is out of bounds of the image. Therefore, isn't added to the window
    			}
    		}
    	}

    	// Sort the pixels in the window
		for (int i = 0; i < effectivePixelsWithinWindow; i++) {
			for (int j = i + 1; j < effectivePixelsWithinWindow; j++) {
				if (window[i] > window[j]) {
					char tmp = window[i];
					window[i] = window[j];
					window[j] = tmp;
				}
			}
		}

    	outputImage[row * numberColumns + column] = window[effectivePixelsWithinWindow / 2]; // Select median pixel value
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
	cv::Mat inputImage = cv::imread(inputImageFileName, IMREAD_GRAYSCALE);
	cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);

    // Allocate GPU memory
    unsigned char *d_inputImage;
    unsigned char *d_outputImage;

    cudaMalloc(&d_inputImage, sizeof(unsigned char) * inputImage.rows * inputImage.cols);
    cudaMalloc(&d_outputImage, sizeof(unsigned char) * inputImage.rows * inputImage.cols);

	cudaMemcpy(d_inputImage, (unsigned char*) inputImage.ptr<unsigned char>(0), sizeof(unsigned char) * inputImage.rows * inputImage.cols, cudaMemcpyHostToDevice);

	int threadsPerBlockPerDim = 16;
	int gridDimx = (inputImage.cols + threadsPerBlockPerDim - 1) / threadsPerBlockPerDim;
	int gridDimy = (inputImage.rows + threadsPerBlockPerDim - 1) / threadsPerBlockPerDim;

	dim3 blockDim(threadsPerBlockPerDim, threadsPerBlockPerDim);
	dim3 gridDim(gridDimx,gridDimy);

	denoiseGPU <<< gridDim, blockDim >>> (d_inputImage, d_outputImage, inputImage.rows, inputImage.cols);

	cudaMemcpy((unsigned char*) outputImage.ptr<unsigned char>(0), d_outputImage, sizeof(unsigned char) * inputImage.rows * inputImage.cols, cudaMemcpyDeviceToHost);

    cv::imwrite(outputImageFileName, outputImage);

    // Free Memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

    return 0;
}
