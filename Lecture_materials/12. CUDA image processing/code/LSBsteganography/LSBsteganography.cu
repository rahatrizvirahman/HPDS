#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

__global__ void hideSecret(uchar3 *originalImage, uchar3 *secretImage, uchar3 *originalsecretImage, int numberRowsSecretImage, int numberColumnsSecretImage)
{
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int    row = blockIdx.y * blockDim.y + threadIdx.y;

	if(column < numberColumnsSecretImage && row < numberRowsSecretImage)
	{
		int tid = row * numberColumnsSecretImage + column;
		int pixel = 8 * tid;

		// Hiding the B channel
		originalsecretImage[pixel + 0].x = originalImage[pixel + 0].x xor ((secretImage[tid].x & 128) >> 7);
		originalsecretImage[pixel + 1].x = originalImage[pixel + 1].x xor ((secretImage[tid].x &  64) >> 6);
		originalsecretImage[pixel + 2].x = originalImage[pixel + 2].x xor ((secretImage[tid].x &  32) >> 5);
		originalsecretImage[pixel + 3].x = originalImage[pixel + 3].x xor ((secretImage[tid].x &  16) >> 4);
		originalsecretImage[pixel + 4].x = originalImage[pixel + 4].x xor ((secretImage[tid].x &   8) >> 3);
		originalsecretImage[pixel + 5].x = originalImage[pixel + 5].x xor ((secretImage[tid].x &   4) >> 2);
		originalsecretImage[pixel + 6].x = originalImage[pixel + 6].x xor ((secretImage[tid].x &   2) >> 1);
		originalsecretImage[pixel + 7].x = originalImage[pixel + 7].x xor ((secretImage[tid].x &   1)     );

		// Hiding the G channel
		originalsecretImage[pixel + 0].y = originalImage[pixel + 0].y xor ((secretImage[tid].y & 128) >> 7);
		originalsecretImage[pixel + 1].y = originalImage[pixel + 1].y xor ((secretImage[tid].y &  64) >> 6);
		originalsecretImage[pixel + 2].y = originalImage[pixel + 2].y xor ((secretImage[tid].y &  32) >> 5);
		originalsecretImage[pixel + 3].y = originalImage[pixel + 3].y xor ((secretImage[tid].y &  16) >> 4);
		originalsecretImage[pixel + 4].y = originalImage[pixel + 4].y xor ((secretImage[tid].y &   8) >> 3);
		originalsecretImage[pixel + 5].y = originalImage[pixel + 5].y xor ((secretImage[tid].y &   4) >> 2);
		originalsecretImage[pixel + 6].y = originalImage[pixel + 6].y xor ((secretImage[tid].y &   2) >> 1);
		originalsecretImage[pixel + 7].y = originalImage[pixel + 7].y xor ((secretImage[tid].y &   1)     );

		// Hiding the R channel
		originalsecretImage[pixel + 0].z = originalImage[pixel + 0].z xor ((secretImage[tid].z & 128) >> 7);
		originalsecretImage[pixel + 1].z = originalImage[pixel + 1].z xor ((secretImage[tid].z &  64) >> 6);
		originalsecretImage[pixel + 2].z = originalImage[pixel + 2].z xor ((secretImage[tid].z &  32) >> 5);
		originalsecretImage[pixel + 3].z = originalImage[pixel + 3].z xor ((secretImage[tid].z &  16) >> 4);
		originalsecretImage[pixel + 4].z = originalImage[pixel + 4].z xor ((secretImage[tid].z &   8) >> 3);
		originalsecretImage[pixel + 5].z = originalImage[pixel + 5].z xor ((secretImage[tid].z &   4) >> 2);
		originalsecretImage[pixel + 6].z = originalImage[pixel + 6].z xor ((secretImage[tid].z &   2) >> 1);
		originalsecretImage[pixel + 7].z = originalImage[pixel + 7].z xor ((secretImage[tid].z &   1)     );
	}
}

__global__ void extractSecret(uchar3 *originalImage, uchar3 *originalsecretImage, uchar3 *extractedsecretImage, int numberRowsSecretImage, int numberColumnsSecretImage)
{
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int    row = blockIdx.y * blockDim.y + threadIdx.y;

	if(column < numberColumnsSecretImage && row < numberRowsSecretImage)
	{
		int tid = row * numberColumnsSecretImage + column;
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

	// Read images
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
		int threadsPerBlockPerDim = 16;
		int gridDimx = (secretImage.cols + threadsPerBlockPerDim - 1) / threadsPerBlockPerDim;
		int gridDimy = (secretImage.rows + threadsPerBlockPerDim - 1) / threadsPerBlockPerDim;

		dim3 blockDim(threadsPerBlockPerDim, threadsPerBlockPerDim);
		dim3 gridDim(gridDimx,gridDimy);

		hideSecret <<< gridDim, blockDim >>> (d_originalImage, d_secretImage, d_originalsecretImage, secretImage.rows, secretImage.cols);

		cudaMemcpy((uchar3*) originalsecretImage.ptr<uchar3>(0), d_originalsecretImage, sizeof(uchar3) * originalImage.rows * originalImage.cols, cudaMemcpyDeviceToHost);

		cv::imwrite(originalsecretFileName, originalsecretImage, {cv::ImwriteFlags::IMWRITE_PNG_COMPRESSION, 0});
	}

	// PART 2
	// Extract secret image from originalsecret with original into extractedsecret

	// Load originalsecret image from file
	originalsecretImage = cv::imread(originalsecretFileName, IMREAD_COLOR);
	cudaMemcpy(d_originalsecretImage, originalsecretImage.ptr<uchar3>(0), sizeof(uchar3) * originalsecretImage.rows * originalsecretImage.cols, cudaMemcpyHostToDevice);

	{
		int threadsPerBlockPerDim = 16;
		int gridDimx = (secretImage.cols + threadsPerBlockPerDim - 1) / threadsPerBlockPerDim;
		int gridDimy = (secretImage.rows + threadsPerBlockPerDim - 1) / threadsPerBlockPerDim;

		dim3 blockDim(threadsPerBlockPerDim, threadsPerBlockPerDim);
		dim3 gridDim(gridDimx,gridDimy);

		extractSecret <<< gridDim, blockDim >>> (d_originalImage, d_originalsecretImage, d_extractedsecretImage, secretImage.rows, secretImage.cols);

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
