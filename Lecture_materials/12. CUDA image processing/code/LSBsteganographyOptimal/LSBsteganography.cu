#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// 32 pixels per block
//  3 channels per pixel
//  8 bits per channel
// 32 * 3 * 8 = 768 threads per block
__global__ void hideSecret(uchar3 *originalImage, uchar3 *secretImage, uchar3 *originalsecretImage, int numberRowsSecretImage, int numberColumnsSecretImage)
{
	int pixelChannel = blockIdx.x * blockDim.x + threadIdx.x; // Identify which pixel and channel this thread is responsible for

	if(pixelChannel < 8 * 3 * (numberRowsSecretImage * numberColumnsSecretImage))
	{
		__shared__ unsigned char subimage[32 * 3]; // 32 pixels x 3 channels

		if(threadIdx.x < (32 * 3)) {
			subimage[threadIdx.x] = *(&secretImage[32*blockIdx.x].x + threadIdx.x); // Load the 32 pixels for the block
		}

		__syncthreads();

		unsigned char bit = (threadIdx.x % 24) / 3; // Identify which bit this thread is responsible for

		// Magic
		*(&originalsecretImage[0].x + pixelChannel) = (*(&originalImage[0].x + pixelChannel)) xor (((*(&subimage[0] + 3*(threadIdx.x/24) + (threadIdx.x % 3))) & (128 >> bit)) >> (7-bit));
	}
}

__global__ void extractSecret(uchar3 *originalImage, uchar3 *originalsecretImage, uchar3 *extractedsecretImage, int numberRowsSecretImage, int numberColumnsSecretImage)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < numberColumnsSecretImage * numberRowsSecretImage)
	{
		int pixel = 8 * tid;

		// Extracting the B channel
		unsigned char x = 0;
		x  = (originalsecretImage[pixel + 0].x xor originalImage[pixel + 0].x) << 7;
		x |= (originalsecretImage[pixel + 1].x xor originalImage[pixel + 1].x) << 6;
		x |= (originalsecretImage[pixel + 2].x xor originalImage[pixel + 2].x) << 5;
		x |= (originalsecretImage[pixel + 3].x xor originalImage[pixel + 3].x) << 4;
		x |= (originalsecretImage[pixel + 4].x xor originalImage[pixel + 4].x) << 3;
		x |= (originalsecretImage[pixel + 5].x xor originalImage[pixel + 5].x) << 2;
		x |= (originalsecretImage[pixel + 6].x xor originalImage[pixel + 6].x) << 1;
		x |= (originalsecretImage[pixel + 7].x xor originalImage[pixel + 7].x);

		// Extracting the G channel
		unsigned char y = 0;
		y  = (originalsecretImage[pixel + 0].y xor originalImage[pixel + 0].y) << 7;
		y |= (originalsecretImage[pixel + 1].y xor originalImage[pixel + 1].y) << 6;
		y |= (originalsecretImage[pixel + 2].y xor originalImage[pixel + 2].y) << 5;
		y |= (originalsecretImage[pixel + 3].y xor originalImage[pixel + 3].y) << 4;
		y |= (originalsecretImage[pixel + 4].y xor originalImage[pixel + 4].y) << 3;
		y |= (originalsecretImage[pixel + 5].y xor originalImage[pixel + 5].y) << 2;
		y |= (originalsecretImage[pixel + 6].y xor originalImage[pixel + 6].y) << 1;
		y |= (originalsecretImage[pixel + 7].y xor originalImage[pixel + 7].y);

		// Extracting the R channel
		unsigned char z = 0;
		z  = (originalsecretImage[pixel + 0].z xor originalImage[pixel + 0].z) << 7;
		z |= (originalsecretImage[pixel + 1].z xor originalImage[pixel + 1].z) << 6;
		z |= (originalsecretImage[pixel + 2].z xor originalImage[pixel + 2].z) << 5;
		z |= (originalsecretImage[pixel + 3].z xor originalImage[pixel + 3].z) << 4;
		z |= (originalsecretImage[pixel + 4].z xor originalImage[pixel + 4].z) << 3;
		z |= (originalsecretImage[pixel + 5].z xor originalImage[pixel + 5].z) << 2;
		z |= (originalsecretImage[pixel + 6].z xor originalImage[pixel + 6].z) << 1;
		z |= (originalsecretImage[pixel + 7].z xor originalImage[pixel + 7].z);

		extractedsecretImage[tid].x = x;
		extractedsecretImage[tid].y = y;
		extractedsecretImage[tid].z = z;
	}
}

bool hasPngExtension(const std::string& fileName) {
    std::string extension = ".png";
    if (fileName.length() >= extension.length()) {
        return (0 == fileName.compare(fileName.length() - extension.length(), extension.length(), extension));
    }
    return false;
}

int main(int argc, char **argv)
{
	if(argc != 5) {
		cerr << "Usage: ./program original.jpg secret.jpg original+secret.png extractedsecret.jpg" << endl;
		return 1;
	}

	string originalImageFileName = string(argv[1]);
	string secretImageFileName = string(argv[2]);
	string originalsecretFileName = string(argv[3]);
	string extractedsecretFileName = string(argv[4]);

	if (!hasPngExtension(originalsecretFileName)) {
        cerr << "Error: " << originalsecretFileName << " does not have a .png extension.\n";
        return 1;
    }

	// Read image
	cv::Mat originalImage = cv::imread(originalImageFileName, IMREAD_COLOR);
	cv::Mat secretImage = cv::imread(secretImageFileName, IMREAD_COLOR);
	// Create output images
	cv::Mat originalsecretImage(originalImage.rows, originalImage.cols, CV_8UC3);
	cv::Mat extractedsecretImage(secretImage.rows, secretImage.cols, CV_8UC3);

	// Original image needs to be at least 8x the size of the secret image to store each secret BGR secrey byte into the LSB of the original image
	if((secretImage.rows * secretImage.cols) > ((originalImage.rows * originalImage.cols) / 8)) {
		cerr << "The secret image is too big to hide in the original image" << endl;
		return 1;
	}

	// Allocate GPU memory
	uchar3 *d_originalImage;
	uchar3 *d_secretImage;
	uchar3 *d_originalsecretImage;
	uchar3 *d_extractedsecretImage;

	cudaMalloc(&d_originalImage, sizeof(uchar3) * originalImage.rows * originalImage.cols);
	cudaMalloc(&d_secretImage, sizeof(uchar3) * secretImage.rows * secretImage.cols);
	cudaMalloc(&d_originalsecretImage, sizeof(uchar3) * originalImage.rows * originalImage.cols);
	cudaMalloc(&d_extractedsecretImage, sizeof(uchar3) * secretImage.rows * secretImage.cols);

	// Copy input images
	cudaMemcpy(d_originalImage, (uchar3*) originalImage.ptr<uchar3>(0), sizeof(uchar3) * originalImage.rows * originalImage.cols, cudaMemcpyHostToDevice);
	cudaMemcpy(d_secretImage, (uchar3*) secretImage.ptr<uchar3>(0), sizeof(uchar3) * secretImage.rows * secretImage.cols, cudaMemcpyHostToDevice);

	// Initialize originalsecret image with original image
	cudaMemcpy(d_originalsecretImage, d_originalImage, sizeof(uchar3) * originalImage.rows * originalImage.cols, cudaMemcpyDeviceToDevice);

	// PART 1
	// Combine original and secret image into originalsecret
	{
		int threadsPerBlock = 32 * 3 * 8; // 32 pixels * 3 channels * 8 bits per pixel
		int gridDim = (secretImage.rows * secretImage.cols * 3 * 8 + threadsPerBlock - 1) / threadsPerBlock;

		hideSecret <<< gridDim, threadsPerBlock >>> (d_originalImage, d_secretImage, d_originalsecretImage, secretImage.rows, secretImage.cols);

		cudaMemcpy((uchar3*) originalsecretImage.ptr<uchar3>(0), d_originalsecretImage, sizeof(uchar3) * originalImage.rows * originalImage.cols, cudaMemcpyDeviceToHost);

		cv::imwrite(originalsecretFileName, originalsecretImage, {cv::ImwriteFlags::IMWRITE_PNG_COMPRESSION, 0});
	}


	// PART 2
	// Extract secret image from originalsecret with original into extractedsecret

	// Load originalsecret image from file
	originalsecretImage = cv::imread(originalsecretFileName, IMREAD_COLOR);
	cudaMemcpy(d_originalsecretImage, originalsecretImage.ptr<uchar3>(0), sizeof(uchar3) * originalsecretImage.rows * originalsecretImage.cols, cudaMemcpyHostToDevice);

	{
		int threadsPerBlock = 32 * 3 * 8; // 32 pixels * 3 channels * 8 bits per pixel
		int gridDim = (secretImage.rows * secretImage.cols + threadsPerBlock - 1) / threadsPerBlock;

		extractSecret <<< gridDim, threadsPerBlock >>> (d_originalImage, d_originalsecretImage, d_extractedsecretImage, secretImage.rows, secretImage.cols);

		cudaMemcpy((uchar3*) extractedsecretImage.ptr<uchar3>(0), d_extractedsecretImage, sizeof(uchar3) * secretImage.rows * secretImage.cols, cudaMemcpyDeviceToHost);

		cv::imwrite(extractedsecretFileName, extractedsecretImage);
	}

	// Free Memory
	cudaFree(d_originalImage);
	cudaFree(d_secretImage);
	cudaFree(d_originalsecretImage);
	cudaFree(d_extractedsecretImage);

    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess)
    {
        fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

	return 0;
}
