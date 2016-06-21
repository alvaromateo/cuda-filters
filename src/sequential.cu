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
	int i, j;
	for(i=0, j=0; i<3*len; i+=3, j++){
		red[j]   = image[i];
		green[j] = image[i+1];
		blue[j]  = image[i+2];
	}

	float gauss[9] = {1./16,2./16,1./16,2./16,4./16,2./16,1./16,2./16,1./16};
	float edge[9] = {0,-1,0,-1,4,-1,0,-1,0}; // NO FUNCIONA, NI TAMPOC SHARPEN
	//Apply filter
	for(i=1; i < height-1; i++){
		for(j=1; j < width-1; j++){
			red[i*width+j] = red[(i-1)*width+(j-1)]*gauss[0] + red[(i-1)*width+(j)]*gauss[1] + red[(i-1)*width+(j+1)]*gauss[2] + red[i*width+(j-1)]*gauss[3] + red[i*width+j]*gauss[4] + red[i*width+(j+1)]*gauss[5] + red[(i+1)*width+(j-1)]*gauss[6] + red[(i+1)*width+j]*gauss[7] + red[(i+1)*width+(j+1)]*gauss[8];
			green[i*width+j] = green[(i-1)*width+(j-1)]*gauss[0] + green[(i-1)*width+(j)]*gauss[1] + green[(i-1)*width+(j+1)]*gauss[2] + green[i*width+(j-1)]*gauss[3] + green[i*width+j]*gauss[4] + green[i*width+(j+1)]*gauss[5] + green[(i+1)*width+(j-1)]*gauss[6] + green[(i+1)*width+j]*gauss[7] + green[(i+1)*width+(j+1)]*gauss[8];
			blue[i*width+j] = blue[(i-1)*width+(j-1)]*gauss[0] + blue[(i-1)*width+(j)]*gauss[1] + blue[(i-1)*width+(j+1)]*gauss[2] + blue[i*width+(j-1)]*gauss[3] + blue[i*width+j]*gauss[4] + blue[i*width+(j+1)]*gauss[5] + blue[(i+1)*width+(j-1)]*gauss[6] + blue[(i+1)*width+j]*gauss[7] + blue[(i+1)*width+(j+1)]*gauss[8];
		}
	}


	for(i=0, j=0; i<3*len; i+=3, j++){
		image[i] = red[j];
		image[i+1] = green[j];
		image[i+2] = blue[j];
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
