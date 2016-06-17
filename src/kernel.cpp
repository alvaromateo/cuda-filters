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


/**
 * Kernel public methods
 */


/*
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
	it = opts.find("color");
	// Initialize type of color
	if (it != opts.end()) {
		this.color = it->second;
	}
	it = opts.find("size");
	// Initialize filter size
	if (it != opts.end()) {
		this.filterSize = it->second;
	}
}

void Kernel::applyFilter(const Filter &filter, Image &image) {
	switch (this.executionType) {
		case sequential:
			sequentialExec(filter, image);
			break;
		case singleCardSyn:
			singleCardSynExec(filter, image);
			break;
		case singleCardAsyn:
			singleCardAsynExec(filter, image);
			break;
		case multiCardSyn:
			multiCardSynExec(filter, image);
			break;
		case multiCardAsyn:
			multiCardAsynExec(filter, image);
			break;
	}
}


/**
 * Kernel private methods
 */


/**
 * Sequential execution to apply the filter to the image. The image is an address of a vector
 * which is consecutively stored in memory (so it behaves as an array), an doesn't have rows
 * or columns strictly speaking. "w"(width) and "h"(height) are the values of the image. The 
 * filter has a fixed size.
 */
void Kernel::sequentialExec(const uchar *filter, const uchar *image, unsigned int w, 
							unsigned int h, uchar *output) {
	// Apply the filter
	for(unsigned int x = 0; x < w; x++) {
		for(unsigned int y = 0; y < h; y++) {
			double red = 0.0, green = 0.0, blue = 0.0;
			// Multiply every value of the filter with corresponding image pixel
			for(int filterX = 0; filterX < filterSize; filterX++) {
				for(int filterY = 0; filterY < filterSize; filterY++) {
					int imageX = (x - filterSize / 2 + filterX + w) % w;
					int imageY = (y - filterSize / 2 + filterY + h) % h;
					red += image[imageY * w + imageX].r * filter[filterY][filterX];
					green += image[imageY * w + imageX].g * filter[filterY][filterX];
					blue += image[imageY * w + imageX].b * filter[filterY][filterX];
				}
			}
			// Truncate values smaller than zero and larger than 255
			output[y * w + x].r = min(max(int(factor * red + bias), 0), 255);
			output[y * w + x].g = min(max(int(factor * green + bias), 0), 255);
			output[y * w + x].b = min(max(int(factor * blue + bias), 0), 255);
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

void Kernel::createEvents(cudaEvent_t &E0, cudaEvent_t &E1, cudaEvent_t &E2, cudaEvent_t &E3) {
	cudaEventCreate(&E0);
	cudaEventCreate(&E1);
	cudaEventCreate(&E2);
	cudaEventCreate(&E3);
}

void Kernel::destroyEvents(cudaEvent_t &E0, cudaEvent_t &E1, cudaEvent_t &E2, cudaEvent_t &E3) {
	cudaEventDestroy(E0);
	cudaEventDestroy(E1);
	cudaEventDestroy(E2);
	cudaEventDestroy(E3);	
}

void Kernel::recordEvent(cudaEvent_t &E) {
	cudaEventRecord(E, 0);
	cudaEventSynchronize(E);
}