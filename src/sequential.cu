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


// Filter definitions
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
    printf("color = %u\n", color);
    printf("bitDepth = %i\n", bitDepth);
    printf("width = %i\n", width);
    printf("height = %i\n", height);

    // bitDepth has the number of channels: 1 for grayscale and 3 for RGB
    uchar **channels = (uchar **) malloc(bitDepth * sizeof(uchar *));
    uchar **output = (uchar **) malloc(bitDepth * sizeof(uchar *));
    
	//Separate the channels
	uint i, j, x;
	uint len = width * height;
	for (x = 0; x < color; ++x) {
		channels[x] = (uchar *) malloc(len * sizeof(uchar));
		output[x] = (uchar *) malloc(len * sizeof(uchar));
		// printf("Pointer to &channel[%u]: %p\n", x, &channels[x]);
		// printf("Pointer to channel[%u]: %p\n", x, channels[x]);
		// printf("Pointer to output[%u]: %p\n", x, output[x]);
	}
	
	for (i = 0, j = 0; i < bitDepth*len; i += bitDepth, ++j){
		for (x = 0; x < color; ++x) { // we leave the alpha channel unchanged
			(channels[x])[j] = image[i + x];
			(output[x])[j] = image[i + x];
			// printf("&(channels[%u])[%u] = %p\n", x, j, &((channels[x])[j]));
			// printf("image[%u + %u] = %u\n", i, x, image[i + x]);
			// printf("(channels[%u])[%u] = %u\n", x, j, (channels[x])[j]);
		}
	}

	// Get filter
	float *filter = arrayFilter[filterType];
	uint filterX, filterY, filterSize;
	// Initialize filterSize
    filterSize = getFiltersize(filterType);

    // Apply filter
    uchar padding = filterSize >> 1; // Divide by 2
    printf("padding = %u\n", padding);
    printf("filterSize = %u\n", filterSize);
	for (i = padding; i < height - padding; ++i) {
		for (j = padding; j < width - padding; ++j) {
			double pixels[color]; // hold the values for each pixel depending on the image channels
			memset(pixels, 0, color);
			for (filterX = 0; filterX < filterSize; ++filterX) {
				for (filterY = 0; filterY < filterSize; ++filterY) {
					uint imageX = (i - padding + filterX);
					uint imageY = (j - padding + filterY);
					for (x = 0; x < color; ++x) {
						pixels[x] = ((double) ((channels[x])[imageX * width + imageY]) * filter[filterX * filterSize + filterY]) + pixels[x];
						// printf("imageX = %u\n", imageX);
						// printf("imageY = %u\n", imageY);
						printf("pixels[%u] = %f * %f\n", x, (double)(channels[x])[imageX * width + imageY], filter[filterX * filterSize + filterY]);
						printf("pixels[%u] = %f\n", x, pixels[x]);
					}
				}
			}
			for (x = 0; x < color; ++x) {
				printf("pixels[%u] = %f\n", x, pixels[x]);
				(output[x])[i * width + j] = (uchar) (pixels[x] < 0) ? 0 : ((pixels[x] > 255) ? 255 : pixels[x]);
				printf("\noutput[%u][%u] -> pixels[%u] = %u\n", i, j, x, (output[x])[i * width + j]);
				printf("image[%u][%u] = %u\n\n\n\n", i, j, (channels[x])[i * width + j]);
			}
		}
	}

	// Print before
	for (x = 0; x < bitDepth; ++x) {
		printf("\n\n\n\n\n");
		for (i = 0; i < width; ++i) {
			for (j = 0; j < height; ++j) {
				printf("%u\t", image[(i * width + j) * bitDepth + x]);
			}
			printf("\n");
		}
	}

	for (i = 0, j = 0; i < bitDepth*len; i += bitDepth, ++j){
		for (x = 0; x < color; ++x) { // we leave the alpha channel unchanged
			image[i + x] = (output[x])[j];
		}
	}

	// Print after
	for (x = 0; x < bitDepth; ++x) {
		printf("\n\n\n\n\n");
		for (i = 0; i < width; ++i) {
			for (j = 0; j < height; ++j) {
				printf("%u\t", image[(i * width + j) * bitDepth + x]);
			}
			printf("\n");
		}
	}

	//Write the image to disk appending "_filter" to its name
	char newImageName[NAME_SIZE] = "\0";
	strncpy(newImageName, imageName, strlen(imageName) - 4);
	// printf(" %d\n",strlen(newImageName));
	strncat(newImageName, "_filter.png", NAME_SIZE - strlen(newImageName) - 1);
	// printf("%s\n",newImageName);
	stbi_write_png(newImageName, width, height, bitDepth, image, width * bitDepth);

    return 0;
}
