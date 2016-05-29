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

void Kernel::sequentialExec(const Filter &filter, Image &image) {
	// Apply the filter
	for(int x = 0; x < w; x++) {
		for(int y = 0; y < h; y++) {
			double red = 0.0, green = 0.0, blue = 0.0;
			// Multiply every value of the filter with corresponding image pixel
			for(int filterY = 0; filterY < filterHeight; filterY++) {
				for(int filterX = 0; filterX < filterWidth; filterX++) {
					int imageX = (x - filterWidth / 2 + filterX + w) % w;
					int imageY = (y - filterHeight / 2 + filterY + h) % h;
					red += image[imageY * w + imageX].r * filter[filterY][filterX];
					green += image[imageY * w + imageX].g * filter[filterY][filterX];
					blue += image[imageY * w + imageX].b * filter[filterY][filterX];
				}
			}
			// Truncate values smaller than zero and larger than 255
			result[y * w + x].r = min(max(int(factor * red + bias), 0), 255);
			result[y * w + x].g = min(max(int(factor * green + bias), 0), 255);
			result[y * w + x].b = min(max(int(factor * blue + bias), 0), 255);
		}
	}
}

void Kernel::singleCardSynExec(const Filter &filter, Image &image) {
	// Variables to calculate time spent in each job
	float TiempoTotal, TiempoKernel;
	cudaEvent_t E0, E1, E2, E3;

	// numero de Blocks en cada dimension 
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

	cudaEventRecord(E0, 0);
	cudaEventSynchronize(E0);

	// Obtener Memoria en el device
	cudaMalloc((float**)&d_A, numBytes); 
	cudaMalloc((float**)&d_B, numBytes); 
	cudaMalloc((float**)&d_C, numBytes); 

	// Copiar datos desde el host en el device 
	cudaMemcpy(d_A, h_A, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, numBytes, cudaMemcpyHostToDevice);

	cudaEventRecord(E1, 0);
	cudaEventSynchronize(E1);

	// Ejecutar el kernel 
	Kernel1<<<dimGrid, dimBlock>>>();

	cudaEventRecord(E2, 0);
	cudaEventSynchronize(E2);

	// Obtener el resultado desde el host 
	cudaMemcpy(h_C, d_C, numBytes, cudaMemcpyDeviceToHost); 

	// Liberar Memoria del device 
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cudaEventRecord(E3, 0);
	cudaEventSynchronize(E3);

	cudaEventElapsedTime(&TiempoTotal,  E0, E3);
	cudaEventElapsedTime(&TiempoKernel, E1, E2);
	
	cudaEventDestroy(E0);
	cudaEventDestroy(E1);
	cudaEventDestroy(E2);
	cudaEventDestroy(E3);
}

void Kernel::singleCardAsynExec(const Filter &filter, Image &image) {

}

void Kernel::multiCardSynExec(const Filter &filter, Image &image) {

}

void Kernel::multiCardAsynExec(const Filter &filter, Image &image) {

}

void Kernel::createEvents(cudaEvent_t &E0, cudaEvent_t &E1, cudaEvent_t &E2, cudaEvent_t &E3) {
	cudaEventCreate(&E0);
	cudaEventCreate(&E1);
	cudaEventCreate(&E2);
	cudaEventCreate(&E3);
}