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

#include <stdio.h>
#include <string.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

float avg3[9] = {1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9};
//float avg5[25] = {1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25};
float sharpenWeak[9] = {0,-1,0,-1,5,-1,0,-1,0};
float sharpenStrong[9] = {-1,-1,-1,-1,9,-1,-1,-1,-1};
float gaussian3[9] = {1./16, 2./16, 1./16, 2./16, 4./16, 2./16, 1./16, 2./16, 1./16};
//float gaussian5[25] = {1./256, 4./256, 6./256, 4./256, 1./256, 4./256, 16./256, 24./256, 16./256, 4./256, 6./256, 24./256, 36./256, 24./256, 6./256, 4./256, 16./256, 24./256, 16./256, 4./256, 1./256, 4./256, 6./256, 4./256, 1./256};
float edgeDetection[9] = {0,1,0,1,-4,1,0,1,0}; //Normalize result by adding 128 to all elements
float embossing[9] = {-2,-1,0,-1,1,1,0,1,2};


int main(int argc, char **argv) {
	char imageName[] = "lena.png"; // by default
    if( argc > 1) {
       // imageName = argv[1];
    }

	int width, height, bitDepth;
	unsigned char* image = stbi_load(imageName, &width, &height, &bitDepth, 3);

    // Check for invalid input
    if( image == NULL ) {
        printf("Could not open or find the image\n");
        return -1;
    }
	//Separate the channels
	int len = width * height;
	unsigned char red[len], green[len], blue[len];
	unsigned char redOut[len], greenOut[len], blueOut[len];
	int i, j;
	for(i=0, j=0; i<3*len; i+=3, j++){
		red[j]   = image[i];
		green[j] = image[i+1];
		blue[j]  = image[i+2];
	}

	//Apply filter
	int filterX, filterY, filterSize=3;
	for(i=1; i < height-1; i++){
		for(j=1; j < width-1; j++){
			double redPixel=0.0, greenPixel=0.0, bluePixel=0.0;
			for(filterY=0; filterY<3; filterY++){
				for(filterX=0; filterX<3; filterX++){
					int imageX = (i - filterSize/2 + filterX);
					int imageY = (j - filterSize/2 + filterY);
					redPixel += red[imageX*width + imageY] * embossing[filterY * filterSize + filterX];
					greenPixel += green[imageX*width + imageY] * embossing[filterY * filterSize + filterX];
					bluePixel += blue[imageX*width + imageY] * embossing[filterY * filterSize + filterX];
				}
			}
			redPixel = (redPixel<0) ? 0 : ((redPixel>255) ? 255 : redPixel);
			greenPixel = (greenPixel<0) ? 0 : ((greenPixel>255) ? 255 : greenPixel);
			bluePixel = (bluePixel<0) ? 0 : ((bluePixel>255) ? 255 : bluePixel);
			redOut[i*width+j] = redPixel;
			greenOut[i*width+j] = greenPixel;
			blueOut[i*width+j] = bluePixel;
		}
	}


	for(i=0, j=0; i<3*len; i+=3, j++){
		image[i] = redOut[j];
		image[i+1] = greenOut[j];
		image[i+2] = blueOut[j];
	}

	//Write the image to disk appending "_filter" to its name
	char newImageName[] = "\0";
	strncpy(newImageName, imageName, strlen(imageName)-4);
//	printf(" %d\n",strlen(newImageName));
	strcat(newImageName, "_filter.png");
//	printf("%s\n",newImageName);
	stbi_write_png(newImageName, width, height, bitDepth, image, width*3);

    return 0;
}
