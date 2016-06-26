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
#include <time.h>

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


int main(int argc, char **argv) {
	// Initialize options
	uchar filterType, threads, pinned;
    char *imageName = getOptions(argc, argv, &filterType, &threads, &pinned);

	int width, height, bitDepth;
	uchar *image = stbi_load(imageName, &width, &height, &bitDepth, 0);

    // Check for invalid input
    if ( image == NULL ) {
        printf("Could not open or find the image\n");
        return -1;
    }

    uint color = !(bitDepth % 2) ? (bitDepth - 1) : bitDepth; // with this we ignore the alpha channel

    // bitDepth has the number of channels: 1 for grayscale and 3 for RGB
    uchar **channels = (uchar **) malloc(color * sizeof(uchar *));
    uchar **output = (uchar **) malloc(color * sizeof(uchar *));
    
	//Separate the channels
	uint i, j, x;
	uint len = width * height;
	for (x = 0; x < color; ++x) {
		channels[x] = (uchar *) malloc(len * sizeof(uchar));
		output[x] = (uchar *) malloc(len * sizeof(uchar));
	}
	
	for (i = 0, j = 0; i < bitDepth*len; i += bitDepth, ++j){
		for (x = 0; x < color; ++x) { // we leave the alpha channel unchanged
			(channels[x])[j] = image[i + x];
			(output[x])[j] = image[i + x];
		}
	}

	// Get filter
	float *filter = arrayFilter[filterType];
	uint filterX, filterY, filterSize;
	// Initialize filterSize
    filterSize = getFiltersize(filterType);

    // Clock to measure time of the execution
    clock_t begin, end;
	double time_spent;
	begin = clock();

    // Apply filter
    uchar padding = filterSize >> 1; // Divide by 2
	for (i = padding; i < height - padding; ++i) {
		for (j = padding; j < width - padding; ++j) {
			float pixels[color]; // hold the values for each pixel depending on the image channels
			memset(pixels, 0, sizeof(pixels));
			for (filterX = 0; filterX < filterSize; ++filterX) {
				for (filterY = 0; filterY < filterSize; ++filterY) {
					uint imageX = (i - padding + filterX);
					uint imageY = (j - padding + filterY);
					for (x = 0; x < color; ++x) {
						pixels[x] = ((float) ((channels[x])[imageX * width + imageY]) * (float) filter[filterX * filterSize + filterY]) + pixels[x];
					}
				}
			}
			for (x = 0; x < color; ++x) {
				(output[x])[i * width + j] = (uchar) (pixels[x] < 0) ? 0 : ((pixels[x] > 255) ? 255 : pixels[x]);
			}
		}
	}

	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("Dimensiones: %dx%dx%d\n", width, height, color);
	printf("Tiempo Global: %4.8f milseg\n", time_spent);

	for (i = 0, j = 0; i < bitDepth*len; i += bitDepth, ++j){
		for (x = 0; x < color; ++x) { // we leave the alpha channel unchanged
			image[i + x] = (output[x])[j];
		}
	}

	// Write the image to disk appending "_filter" to its name
	char newImageName[NAME_SIZE] = "\0";
	strncpy(newImageName, imageName, strlen(imageName) - 4);
	strncat(newImageName, "_filter.png", NAME_SIZE - strlen(newImageName) - 1);
	stbi_write_png(newImageName, width, height, bitDepth, image, width * bitDepth);

    return 0;
}
