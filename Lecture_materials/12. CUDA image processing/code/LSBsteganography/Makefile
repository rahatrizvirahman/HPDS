all: LSBsteganography 
LSBsteganography: LSBsteganography.cu
	nvcc LSBsteganography.cu -o LSBsteganography -I/usr/include/opencv4/ -I/opt/opencv-4.8.1/include/opencv4/ -L/opt/opencv-4.8.1/lib64 -l opencv_core -l opencv_imgproc -l opencv_imgcodecs	
clean:
	rm LSBsteganography