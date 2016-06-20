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
#include <utility>
#include <fstream>


/*
 * Filter definitions
 */

float filter_avg3[9] = {1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9};
float filter_avg5[25] = {1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25};
float filter_sharpenWeak[9] = {0,-1,0,-1,5,-1,0,-1,0};
float filter_sharpenStrong[9] = {-1,-1,-1,-1,9,-1,-1,-1,-1};
float filter_gaussian3[9] = {1./16, 2./16, 1./16, 2./16, 4./16, 2./16, 1./16, 2./16, 1./16};
float filter_gaussian5[25] = {1./256, 4./256, 6./256, 4./256, 1./256, 4./256, 16./256, 24./256, 16./256, 4./256, 6./256, 24./256, 36./256, 24./256, 6./256, 4./256, 16./256, 24./256, 16./256, 4./256, 1./256, 4./256, 6./256, 4./256, 1./256};
float filter_edgeDetection[9] = {0,1,0,1,-4,1,0,1,0}; //Normalize result by adding 128 to all elements
float filter_embossing[9] = {-2,-1,0,-1,1,1,0,1,2};



std::pair<float *, uint> getFilter(uchar filterType) {
	float *filter;
	uint size;
	switch (filterType) {
        case 0:
            filter = &filter_avg3[0];
            size = 3;
            break;
        case 1:
            filter = &filter_avg5[0];
            size = 5;
            break;
        case 2:
            filter = &filter_sharpenWeak[0];
            size = 3;
            break;
        case 3:
            filter = &filter_sharpenStrong[0];
            size = 3;
            break;
        case 4:
            filter = &filter_gaussian3[0];
            size = 3;
            break;
        case 5:
            filter = &filter_gaussian5[0];
            size = 5;
            break;
        case 6:
            filter = &filter_edgeDetection[0];
            size = 3;
            break;
        case 7:
            filter = &filter_embossing[0];
            size = 3;
            break;
    }
    return std::make_pair(filter, size);
}


/**
 * Kernels
 */

__global__ void kernel(uint filterSize, uint margin, uint imgWidth, uint imgHeight, float *filter, uchar *img, uchar *res) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < imgWidth - margin) && (i > margin) && (j < imgHeight - margin) && (j > margin)) {
		// Multiply every value of the filter with corresponding image pixel
		float val = 0.0;
		for(int filterY = 0; filterY < filterSize; filterY++) {
			for(int filterX = 0; filterX < filterSize; filterX++) {
				int imageX = (i - margin + filterX);
				int imageY = (j - margin + filterY);
				val += img[imageY * imgWidth + imageX] * filter[filterY * filterSize + filterX];
			}
		}
		// Output value
		uchar out = (uchar) (val > 0 ? val : 0);
		res[j * imgWidth + i] = (uchar) (out < 255 ? out : 255);
	}
}


/**
 * CUDA util methods
 */

void recordEvent(cudaEvent_t &E) {
	cudaEventRecord(E, 0);
	cudaEventSynchronize(E);
}

template < typename U >
void copyMatrix(const U *matrix, U *mat, uint len) {
	for (uint i = 0; i < len; ++i) {
		mat[i] = matrix[i];
	}
}


/*
 * Test methods
 */

/**
 * CommandLineParserTest methods
 */

void printImage(const uchar *m, uint w, uint h, const std::string &fileName) {
	std::ofstream myfile;
	myfile.open (fileName.c_str());
	for (int i = 0; i < w; ++i) {
		for (int j = 0; j < h; ++j) {
			myfile << m[i * h + j] << " ";
		}
		myfile << std::endl;
	}
}

void printFilter(const float *m, uint w, uint h, const std::string &fileName) {
	std::ofstream myfile;
	myfile.open (fileName.c_str());
	for (int i = 0; i < w; ++i) {
		for (int j = 0; j < h; ++j) {
			myfile << m[i * h + j] << " ";
		}
		myfile << std::endl;
	}
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
		this->executionType = it->second;
	}
	it = opts.find("pinned");
	// Initialize type of memory
	if (it != opts.end()) {
		this->pinned = it->second;
	}
	it = opts.find("threads");
	// Initialize number of threads per block
	if (it != opts.end()) {
		this->nThreads = it->second;
	}
	it = opts.find("filter");
	// Initialize filter type
	if (it != opts.end()) {
		this->filterType = it->second;
	}
	
	this->imageNames = clp.getImages();

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
		switch (this->executionType) {
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

void Kernel::saveImages() {
	for (int i = 0; i < images.size(); ++i) {
		images[i].saveImageToDisk();
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
	std::vector<Image> imgs;
	for (int i = 0; i < imageNames.size(); ++i) {
		Image image(imageNames[i]);
		imgs.push_back(image);
#if DEBUG
    	std::cout << "kernel::loadImages width: " << imgs[i].getWidth() << std::endl;
#endif
	}
	return imgs;
}

/**
 * Method to initialize the filter and save it into a private variable.
 */
Matrix<float> Kernel::initFilter() {
	std::pair<float *, uint> filterOpts = getFilter(filterType);
	Matrix<float> filter(filterOpts.first, filterOpts.second, filterOpts.second);
	return filter;
}

/**
 * Sequential execution to apply the filter to the image. The image is an address of a vector
 * which is consecutively stored in memory (so it behaves as an array), an doesn't have rows
 * or columns strictly speaking. "w"(width) and "h"(height) are the values of the image. The 
 * filter has a fixed size.
 */
void Kernel::sequentialExec(const Matrix<float> &f, Image &image) {
	// Initialize the values
	float *filter = f.getMatrix();
	Image output(image);
	uint w, h, filterSize;
	w = image.getWidth();
	h = image.getHeight();
	filterSize = f.getWidth(); // In the filter width == height

	// Print options for debugging
#if DEBUG
	std::cerr << "Image width: " << w << std::endl;
	std::cerr << "Image height: " << h << std::endl;
	printFilter(filter, f.getWidth(), f.getHeight(), "build/filter.txt");
	printImage(image[0].getMatrix(), image.getWidth(), image.getHeight(), "build/red.txt");
#endif
	
	// Apply the filter
	for(unsigned int x = 0; x < w; x++) {
		for(unsigned int y = 0; y < h; y++) {
			double red = 0.0, green = 0.0, blue = 0.0;
			// Multiply every value of the filter with corresponding image pixel
			for(int filterY = 0; filterY < filterSize; filterY++) {
				for(int filterX = 0; filterX < filterSize; filterX++) {
					int imageX = (x - filterSize / 2 + filterX);
					int imageY = (y - filterSize / 2 + filterY);
					red += image[0][imageY * w + imageX] * filter[filterY * filterSize + filterX];
					green += image[1][imageY * w + imageX] * filter[filterY * filterSize + filterX];
					blue += image[2][imageY * w + imageX] * filter[filterY * filterSize + filterX];
				}
			}
			// Truncate values smaller than zero and larger than 255
			output[0][y * w + x] = std::min(std::max(int(red), 0), 255);
			output[1][y * w + x] = std::min(std::max(int(green), 0), 255);
			output[2][y * w + x] = std::min(std::max(int(blue), 0), 255);
		}
	}
	// Copy the result
	image[0].setMatrix(output[0].getMatrix());
	image[1].setMatrix(output[1].getMatrix());
	image[2].setMatrix(output[2].getMatrix());

}

void Kernel::singleCardSynExec(const Matrix<float> &filter, Image &image) {
	// Variables to calculate time spent in each job
	float TiempoTotal, TiempoKernel;
	cudaEvent_t E0, E1, E2, E3;
	// Pointers to variables in the device
	float *f;
	uchar *iRed, *iGreen, *iBlue, *iModRed, *iModGreen, *iModBlue;
	// Pointers to variables in the host
	float *f_H;
	uchar *iRed_H, *iGreen_H, *iBlue_H, *iModRed_H, *iModGreen_H, *iModBlue_H;


	// Number of blocks in each dimension 
	uint nBlocksX = (image.getWidth() + nThreads - 1) / nThreads; 
	uint nBlocksY = (image.getHeight() + nThreads - 1) / nThreads;

	uint numBytesImage = image.getWidth() * image.getHeight() * sizeof(uchar);
	uint numBytesFilter = filter.getWidth() * filter.getHeight() * sizeof(float);
	std::cerr << "numBytesImage = " << numBytesImage << std::endl;
	std::cerr << "numBytesFilter = " << numBytesFilter << std::endl;
	std::cerr << "nBlocksX = " << nBlocksX << std::endl;
	std::cerr << "nBlocksY = " << nBlocksY << std::endl;
	std::cerr << "Image Width = " << image.getWidth() << std::endl;
	std::cerr << "Image Height = " << image.getHeight() << std::endl;

	dim3 dimGrid(nBlocksX, nBlocksY, 1);
	dim3 dimBlock(nThreads, nThreads, 1);

	cudaEventCreate(&E0);
	cudaEventCreate(&E1);
	cudaEventCreate(&E2);
	cudaEventCreate(&E3);

	if (pinned) {
		// memory for the filter
		cudaMallocHost((float**)&f_H, numBytesFilter); 
		// memory for the image
	    cudaMallocHost((uchar**)&iRed_H, numBytesImage); 
	    cudaMallocHost((uchar**)&iGreen_H, numBytesImage);
	    cudaMallocHost((uchar**)&iBlue_H, numBytesImage);
	    // memory for the result
	    cudaMallocHost((uchar**)&iModRed_H, numBytesImage);
	    cudaMallocHost((uchar**)&iModGreen_H, numBytesImage);
	    cudaMallocHost((uchar**)&iModBlue_H, numBytesImage);
	} else {
		// memory for the filter
		f_H = (float*) malloc(numBytesFilter); 
		// memory for the image
	    iRed_H = (uchar*) malloc(numBytesImage); 
	    iGreen_H = (uchar*) malloc(numBytesImage); 
	    iBlue_H = (uchar*) malloc(numBytesImage); 
	    // memory for the result
	    iModRed_H = (uchar*) malloc(numBytesImage);
	    iModGreen_H = (uchar*) malloc(numBytesImage);
	    iModBlue_H = (uchar*) malloc(numBytesImage);
	}

	// Initialize matrixes
	copyMatrix(filter.getMatrix(), f_H, numBytesFilter);
	copyMatrix(image[0].getMatrix(), iRed_H, numBytesImage);
	copyMatrix(image[1].getMatrix(), iGreen_H, numBytesImage);
	copyMatrix(image[2].getMatrix(), iBlue_H, numBytesImage);

	//recordEvent(E0);
	cudaEventRecord(E0, 0);
	cudaEventSynchronize(E0);

	// Get memory in device
	// filter
	cudaMalloc((float**)&f, numBytesFilter); 
	// image
	cudaMalloc((uchar**)&iRed, numBytesImage); 
	cudaMalloc((uchar**)&iGreen, numBytesImage); 
	cudaMalloc((uchar**)&iBlue, numBytesImage); 
	// modified image
	cudaMalloc((uchar**)&iModRed, numBytesImage); 
	cudaMalloc((uchar**)&iModGreen, numBytesImage); 
	cudaMalloc((uchar**)&iModBlue, numBytesImage); 

	// Copy data from host to device 
	cudaMemcpy(f, f_H, numBytesFilter, cudaMemcpyHostToDevice);
	cudaMemcpy(iRed, iRed_H, numBytesImage, cudaMemcpyHostToDevice);
	cudaMemcpy(iGreen, iRed_H, numBytesImage, cudaMemcpyHostToDevice);
	cudaMemcpy(iBlue, iRed_H, numBytesImage, cudaMemcpyHostToDevice);

	//recordEvent(E1);
	cudaEventRecord(E1, 0);
	cudaEventSynchronize(E1);

	// Execute the kernel
	kernel<<<dimGrid, dimBlock>>>(filter.getWidth(), filter.getWidth() / 2, image.getWidth(), image.getHeight(), f, iRed, iModRed);
	kernel<<<dimGrid, dimBlock>>>(filter.getWidth(), filter.getWidth() / 2, image.getWidth(), image.getHeight(), f, iGreen, iModGreen);
	kernel<<<dimGrid, dimBlock>>>(filter.getWidth(), filter.getWidth() / 2, image.getWidth(), image.getHeight(), f, iBlue, iModBlue);

	//recordEvent(E2);
	cudaEventRecord(E2, 0);
	cudaEventSynchronize(E2);

	// Get the result to the host 
	cudaMemcpy(iModRed_H, iModRed, numBytesImage, cudaMemcpyDeviceToHost); 
	cudaMemcpy(iModGreen_H, iModGreen, numBytesImage, cudaMemcpyDeviceToHost);
	cudaMemcpy(iModBlue_H, iModBlue, numBytesImage, cudaMemcpyDeviceToHost);

	// Copy the result to image
	image[0].setMatrix(iModRed_H);
	image[1].setMatrix(iModGreen_H);
	image[2].setMatrix(iModBlue_H);

	// Free memory of the device 
	cudaFree(f);
	cudaFree(iRed);
	cudaFree(iGreen);
	cudaFree(iBlue);
	cudaFree(iModRed);
	cudaFree(iModGreen);
	cudaFree(iModBlue);

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

	if (pinned) {
		cudaFreeHost(f_H);
		cudaFreeHost(iRed_H);
		cudaFreeHost(iGreen_H);
		cudaFreeHost(iBlue_H);
		cudaFreeHost(iModRed_H);
		cudaFreeHost(iModGreen_H);
		cudaFreeHost(iModBlue_H);
	} else {
		free(f_H);
		free(iRed_H);
		free(iGreen_H);
		free(iBlue_H);
		free(iModRed_H);
		free(iModGreen_H);
		free(iModBlue_H);
	}
}

void Kernel::singleCardAsynExec(const Matrix<float> &f, Image &image) {

}

void Kernel::multiCardSynExec(const Matrix<float> &f, Image &image) {

}

void Kernel::multiCardAsynExec(const Matrix<float> &f, Image &image) {

}
