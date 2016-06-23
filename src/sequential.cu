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
    // bitDepth has the number of channels: 1 for grayscale and 3 for RGB
    // TODO: check number of channels to apply the filter correctly if the image is in grayscale

	//Separate the channels
	uint len = width * height;
	uchar red[len], green[len], blue[len];
	uchar redOut[len], greenOut[len], blueOut[len];
	uint i, j;

	for (i = 0, j = 0; i < 3*len; i += 3, j++){
		red[j]   = image[i];
		green[j] = image[i+1];
		blue[j]  = image[i+2];
	}

	// Get filter
	float *filter = arrayFilter[filterType];
	uint filterX, filterY, filterSize;
	// Initialize filterSize
    filterSize = getFiltersize(filterType);

    // Apply filter
    uchar padding = filterSize >> 1; // Divide by 2
	for (i = padding; i < height - padding; i++) {
		for j = padding; j < width - padding; j++) {
			double redPixel=0.0, greenPixel=0.0, bluePixel=0.0;
			for (filterY = 0; filterY < filterSize; filterY++) {
				for (filterX=0; filterX < filterSize; filterX++) {
					uint imageX = (i - padding + filterX);
					uint imageY = (j - padding + filterY);
					redPixel += red[imageX * width + imageY] * filter[filterY * filterSize + filterX];
					greenPixel += green[imageX * width + imageY] * filter[filterY * filterSize + filterX];
					bluePixel += blue[imageX * width + imageY] * filter[filterY * filterSize + filterX];
				}
			}
			redPixel = (redPixel < 0) ? 0 : ((redPixel > 255) ? 255 : redPixel);
			greenPixel = (greenPixel < 0) ? 0 : ((greenPixel > 255) ? 255 : greenPixel);
			bluePixel = (bluePixel < 0) ? 0 : ((bluePixel > 255) ? 255 : bluePixel);
			redOut[i * width + j] = redPixel;
			greenOut[i * width + j] = greenPixel;
			blueOut[i * width + j] = bluePixel;
		}
	}


	for (i = 0, j = 0; i < 3*len; i += 3, j++) {
		image[i] = redOut[j];
		image[i+1] = greenOut[j];
		image[i+2] = blueOut[j];
	}

	//Write the image to disk appending "_filter" to its name
	char newImageName[NAME_SIZE] = "\0";
	strncpy(newImageName, imageName, strlen(imageName) - 4);
	strncat(newImageName, "_filter.png", NAME_SIZE - strlen(newImageName) - 1);
	stbi_write_png(newImageName, width, height, bitDepth, image, width * 3);

    return 0;
}
