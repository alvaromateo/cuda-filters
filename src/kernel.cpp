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
#include "kernel.h"
#include <algorithm>


/**
 * Kernels
 */

__global__ void kernel1() {

}


/**
 * CUDA util methods
 */

void createEvents(cudaEvent_t &E0, cudaEvent_t &E1, cudaEvent_t &E2, cudaEvent_t &E3) {
	cudaEventCreate(&E0);
	cudaEventCreate(&E1);
	cudaEventCreate(&E2);
	cudaEventCreate(&E3);
}

void destroyEvents(cudaEvent_t &E0, cudaEvent_t &E1, cudaEvent_t &E2, cudaEvent_t &E3) {
	cudaEventDestroy(E0);
	cudaEventDestroy(E1);
	cudaEventDestroy(E2);
	cudaEventDestroy(E3);	
}

void recordEvent(cudaEvent_t &E) {
	cudaEventRecord(E, 0);
	cudaEventSynchronize(E);
}


/*
 * Kernel public methods
 */


/**
 * Constructor that takes a CommandLineParser object and uses it to initialize
 * all the values needed for the kernel to be launched.
 */
Kernel::Kernel(const CommandLineParser &clp) {
	std::map<std::string, unsigned short> opts = clp.getOptions();
	std::map<std::string, unsigned short>::const_iterator it = opts.find("exec");
	// Initialize execution type
	if (it != opts.end()) {
		this.executionType = it->second;
	}
	it = opts.find("pinned");
	// Initialize type of memory
	if (it != opts.end()) {
		this.pinned = it->second;
	}
	it = opts.find("threads");
	// Initialize number of threads per block
	if (it != opts.end()) {
		this.nThreads = it->second;
	}
	it = opts.find("size");
	// Initialize filter size
	if (it != opts.end()) {
		this.filterSize = it->second;
	}
	this.imageNames = clp.getImages();

	/* For the moment not used
	it = opts.find("color");
	// Initialize type of color
	if (it != opts.end()) {
		this.color = it->second;
	}
	*/
}

void Kernel::applyFilter() {
	images = loadImages();
	filter = initFilter();
	for (int i = 0; i < images.size(); ++i) {
		switch (this.executionType) {
			case sequential:
				sequentialExec(filter, images[i]);
				break;
			case singleCardSyn:
				singleCardSynExec(filter, images[i]);
				break;
			case singleCardAsyn:
				singleCardAsynExec(filter, images[i]);
				break;
			case multiCardSyn:
				multiCardSynExec(filter, images[i]);
				break;
			case multiCardAsyn:
				multiCardAsynExec(filter, images[i]);
				break;
		}
	}
}


/**
 * Kernel private methods
 */


/**
 * This method loads the images based on the values of the CommandLineParser. They are
 * stored in a vector containing all the images inserted.
 */
std::vector<Image> Kernel::loadImages() {

}

/**
 * Method to initialize the filter and save it into a private variable.
 */
Matrix Kernel::initFilter() {

}

/**
 * Sequential execution to apply the filter to the image. The image is an address of a vector
 * which is consecutively stored in memory (so it behaves as an array), an doesn't have rows
 * or columns strictly speaking. "w"(width) and "h"(height) are the values of the image. The 
 * filter has a fixed size.
 */
void Kernel::sequentialExec(const uchar *filter, const uchar *image, unsigned int w, 
							unsigned int h, uchar *output, unsigned int filterSize) {
	// Initialize the values

	// Apply the filter
	for(unsigned int x = 0; x < w; x++) {
		for(unsigned int y = 0; y < h; y++) {
			double red = 0.0, green = 0.0, blue = 0.0;
			// Multiply every value of the filter with corresponding image pixel
			for(int filterY = 0; filterY < filterSize; filterY++) {
				for(int filterX = 0; filterX < filterSize; filterX++) {
					int imageX = (x - filterSize / 2 + filterX + w) % w;
					int imageY = (y - filterSize / 2 + filterY + h) % h;
					red += image.getImg()[0].getMatrix()[imageY * w + imageX] * filter[filterY * filterSize + filterX];
					green += image.getImg()[1].getMatrix()[imageY * w + imageX] * filter[filterY * filterSize + filterX];
					blue += image.getImg()[2].getMatrix()[imageY * w + imageX] * filter[filterY * filterSize + filterX];
				}
			}
			// Truncate values smaller than zero and larger than 255
			output[y * w + x].r = std::min(std::max(int(factor * red + bias), 0), 255);
			output[y * w + x].g = std::min(std::max(int(factor * green + bias), 0), 255);
			output[y * w + x].b = std::min(std::max(int(factor * blue + bias), 0), 255);
		}
	}
}

void Kernel::singleCardSynExec(const uchar *filter, uchar *image, unsigned int imageSize) {
	// Variables to calculate time spent in each job
	float TiempoTotal, TiempoKernel;
	cudaEvent_t E0, E1, E2, E3;

	// Number of blocks in each dimension 
	unsigned int nBlocksX = image.getSizeX / nThreads; 
	unsigned int nBlocksY = image.getSizeY / nThreads;

	unsigned int numBytes = image.getSizeX * image.getSizeY * sizeof(uchar);

	dim3 dimGrid(nBlocksX, nBlocksY, 1);
	dim3 dimBlock(nThreads, nThreads, 1);

	createEvents(&E0, &E1, &E2, &E3);

	if (pinned) {
		getPinnedMemory();
	} else {
		getMemory();
	}

	recordEvent(E0);

	// Get memory in device
	cudaMalloc((float**)&d_A, numBytes); 
	cudaMalloc((float**)&d_B, numBytes); 
	cudaMalloc((float**)&d_C, numBytes); 

	// Copy data from host to device 
	cudaMemcpy(d_A, h_A, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, numBytes, cudaMemcpyHostToDevice);

	recordEvent(E1);

	// Execute the kernel
	Kernel1<<<dimGrid, dimBlock>>>();

	recordEvent(E2);

	// Get the result to the host 
	cudaMemcpy(h_C, d_C, numBytes, cudaMemcpyDeviceToHost); 

	// Free memory of the device 
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	recordEvent(E3);

	cudaEventElapsedTime(&TiempoTotal,  E0, E3);
	cudaEventElapsedTime(&TiempoKernel, E1, E2);

	// Print results. TODO

	if (pinned) {
		freePinnedMemory();
	} else {
		freeMemory();
	}

	destroyEvents(&E0, &E1, &E2, &E3);
	
}

void Kernel::singleCardAsynExec(const uchar *filter, uchar *image, unsigned int imageSize) {

}

void Kernel::multiCardSynExec(const uchar *filter, uchar *image, unsigned int imageSize) {

}

void Kernel::multiCardAsynExec(const uchar *filter, uchar *image, unsigned int imageSize) {

}
