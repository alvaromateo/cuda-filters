/*

**************************************************************************

cuda-filters == command line program to apply various filters to images
Copyright (C) 2016  Alvaro Mateo (alvaromateo9@gmail.com)
					Biel Pieras (bpierasmorell@gmail.com)

**************************************************************************

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
  
***************************************************************************

*/

// Includes
#include <math.h>

extern "C" {
	#include "readCommandLine.h"
}

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

float avg3[9] = {1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9};
float avg5[25] = {1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25};
float sharpenWeak[9] = {0,-1,0,-1,5,-1,0,-1,0};
float sharpenStrong[9] = {-1,-1,-1,-1,9,-1,-1,-1,-1};
float gaussian3[9] = {1./16, 2./16, 1./16, 2./16, 4./16, 2./16, 1./16, 2./16, 1./16};
float gaussian5[25] = {1./256, 4./256, 6./256, 4./256, 1./256, 4./256, 16./256, 24./256, 16./256, 4./256, 6./256, 24./256, 36./256, 24./256, 6./256, 4./256, 16./256, 24./256, 16./256, 4./256, 1./256, 4./256, 6./256, 4./256, 1./256};
float edgeDetection[9] = {0,1,0,1,-4,1,0,1,0}; //Normalize result by adding 128 to all elements
float embossing[9] = {-2,-1,0,-1,1,1,0,1,2};

// Filter array
float *arrayFilter[] = {&avg3[0], &avg5[0], &sharpenWeak[0], &sharpenStrong[0], &gaussian3[0], &gaussian5[0], &edgeDetection[0], &embossing[0]};

// Methods
uchar getFiltersize(uchar filterType) {
	uchar filterSize = 3;
	switch (filterType) {
		case 1:
		case 5:
			filterSize = 5;
			break;
	}
	return filterSize;
}

void initFilter(float *filter, uint filterSize, uchar filterType) {
	for (uint a = 0; a < filterSize; ++a) {
		filter[a] = (arrayFilter[filterType])[a];
	}
}

void CheckCudaError(char sms[], int line) {
	cudaError_t error;
	error = cudaGetLastError();
	if (error) {
		printf("(ERROR) %s - %s in %s at line %d\n", sms, cudaGetErrorString(error), __FILE__, line);
		exit(EXIT_FAILURE);
	}
}

__global__ void kernel(int width, int height, int filterSize, float *filt, uchar *img, uchar *out) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	uint j = blockIdx.y * blockDim.y + threadIdx.y;
	uint padding = filterSize / 2;
	unsigned long int index = i * width + j;

	if ((i > padding) && (j > padding) && (i < width - padding) && (j < height - padding)) {
		float tmp = 0.0;
		for (uint filterX = 0; filterX < filterSize; ++filterX) {
			for (uint filterY = 0; filterY < filterSize; ++filterY) {
				uint imageX = (i - padding + filterX);
				uint imageY = (j - padding + filterY);
				tmp += ((float) img[imageX * width + imageY] * (float) filt[filterX * filterSize + filterY]);
			}
		}
		out[index] = (uchar) (tmp < 0) ? 0 : ((tmp > 255) ? 255 : tmp);
	}
}


int main(int argc, char **argv) {
	// Initialize options
	uchar filterType, threads, pinned;
    char *imageName = getOptions(argc, argv, &filterType, &threads, &pinned);

    // bitDepth has the number of channels: 1 for grayscale and 3 for RGB
	int width, height, bitDepth;
	uchar *image = stbi_load(imageName, &width, &height, &bitDepth, 0);

    // Check for invalid input
    if ( image == NULL ) {
        printf("Could not open or find the image\n");
        return -1;
    }

    uint color = !(bitDepth % 2) ? (bitDepth - 1) : bitDepth; // with this we ignore the alpha channel

	/*
	 * Start kernel part!
	 */

	// Pointers to variables in the host
    uchar **channels = (uchar **) malloc(color * sizeof(uchar *));
    // Pointers to variables in the device
    uchar **channelsDevice = (uchar **) malloc(color * sizeof(uchar *));
    uchar **outputDevice = (uchar **) malloc(color * sizeof(uchar *));
    
	//Separate the channels
	uint i, j, x;
	uint len = width * height;
	uint numBytesImage = len * sizeof(uchar);

	for (x = 0; x < color; ++x) {
		if (pinned) {
			cudaMallocHost((uchar **) &channels[x], numBytesImage);
			CheckCudaError((char *) "Obtener Memoria en el host", __LINE__);
		} else {
			channels[x] = (uchar *) malloc(len * sizeof(uchar));
		}
	}
	
	// Initialize matrixs
	for (i = 0, j = 0; i < bitDepth*len; i += bitDepth, ++j){
		for (x = 0; x < color; ++x) { // we leave the alpha channel unchanged
			(channels[x])[j] = image[i + x];
		}
	}

	// Get filter
	float *filter, *filterDevice;
	uint filterSize, numBytesFilter;

	// Initialize filterSize
    filterSize = getFiltersize(filterType);
    numBytesFilter = filterSize * filterSize * sizeof(float);

	if (pinned) {
		cudaMallocHost((float **) &filter, numBytesFilter);
		CheckCudaError((char *) "Obtener Memoria en el host", __LINE__);
	} else {
		filter = (float *) malloc(filterSize * sizeof(float));
	}
	initFilter(filter, filterSize, filterType);

    // Variables to calculate time spent in each job
	float TiempoTotal, TiempoKernel;
	cudaEvent_t E0, E1, E2, E3;

	// Number of blocks in each dimension 
	uint nBlocksX = (width + threads - 1) / threads; 
	uint nBlocksY = (height + threads - 1) / threads;

	dim3 dimGrid(nBlocksX, nBlocksY, 1);
	dim3 dimBlock(threads, threads, 1);

	cudaEventCreate(&E0);
	cudaEventCreate(&E1);
	cudaEventCreate(&E2);
	cudaEventCreate(&E3);

	cudaEventRecord(E0, 0);
	cudaEventSynchronize(E0);

	// Get memory in device and send data
	// Filter
	cudaMalloc((float**) &filterDevice, numBytesFilter); 
	CheckCudaError((char *) "Obtener Memoria en el device", __LINE__);
	cudaMemcpy(filterDevice, filter, numBytesFilter, cudaMemcpyHostToDevice);
	CheckCudaError((char *) "Copiar Memoria en el device", __LINE__);
	// Image
	for (x = 0; x < color; ++x) {
		cudaMalloc((uchar **) &channelsDevice[x], numBytesImage);
		CheckCudaError((char *) "Obtener Memoria en el device", __LINE__);
		cudaMalloc((uchar **) &outputDevice[x], numBytesImage);
		CheckCudaError((char *) "Obtener Memoria en el device", __LINE__);
		cudaMemcpy(channelsDevice[x], channels[x], numBytesImage, cudaMemcpyHostToDevice);
		CheckCudaError((char *) "Copiar Memoria en el device", __LINE__);
	}

	cudaEventRecord(E1, 0);
	cudaEventSynchronize(E1);

	// Execute the kernel
	for (x = 0; x < color; ++x) {
		kernel<<<dimGrid, dimBlock>>>(width, height, filterSize, filterDevice, channelsDevice[x], outputDevice[x]);
	}
	printf("After the kernel\n");
	
	//recordEvent(E2);
	cudaEventRecord(E2, 0);
	cudaEventSynchronize(E2);

	// Get the result to the host and free memory
	cudaFree(filterDevice);
	CheckCudaError((char *) "Liberar Memoria en el device", __LINE__);
	for (x = 0; x < color; ++x) {
		cudaMemcpy(channels[x], outputDevice[x], numBytesImage, cudaMemcpyDeviceToHost);
		CheckCudaError((char *) "Copiar Memoria en el host", __LINE__);
		cudaFree(channelsDevice[x]);
		CheckCudaError((char *) "Liberar Memoria en el device", __LINE__);
		cudaFree(outputDevice[x]);
		CheckCudaError((char *) "Liberar Memoria en el device", __LINE__);
	}

	//recordEvent(E3);
	cudaEventRecord(E3, 0);
	cudaEventSynchronize(E3);

	cudaEventElapsedTime(&TiempoTotal,  E0, E3);
	cudaEventElapsedTime(&TiempoKernel, E1, E2);

	// Print results. TODO

	cudaEventDestroy(E0);
	cudaEventDestroy(E1);
	cudaEventDestroy(E2);
	cudaEventDestroy(E3);

	// Rejoin the channels to save the image
    for (i = 0, j = 0; i < bitDepth*len; i += bitDepth, ++j){
		for (x = 0; x < color; ++x) { // we leave the alpha channel unchanged
			image[i + x] = (channels[x])[j];
		}
	}

	// Free memory of the host
	if (pinned) {
		cudaFreeHost(filter);
	} else {
		free(filter);
	}
	for (x = 0; x < color; ++x) {
		if (pinned) {
			cudaFreeHost(channels[x]);
		} else {
			free(channels[x]);
		}
	}
	free(channels);
	free(channelsDevice);
	free(outputDevice);

	/*
	 * End kernel part!
	 */


	// Write the image to disk appending "_filter" to its name
	char newImageName[NAME_SIZE] = "\0";
	strncpy(newImageName, imageName, strlen(imageName) - 4);
	strncat(newImageName, "_filter.png", NAME_SIZE - strlen(newImageName) - 1);
	stbi_write_png(newImageName, width, height, bitDepth, image, width * bitDepth);

    return 0;
}
