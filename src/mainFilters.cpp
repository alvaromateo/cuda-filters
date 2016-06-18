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
#include <iostream>
#include <string>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "tools.h"
#include "test.h"

using namespace std;

// Change DEBUG to 0 to disable debugging
#define DEBUG 1


int main(int argc, char **argv) {
	CommandLineParser clp(argc, argv); // read commandline options (tools.h)

#if DEBUG
	CommandLineParserTest clpTest(&clp);
	clpTest.doTest();
#endif
	
	// initialize filter
	unsigned int filterSize = clp.getFilterSize();
	MATRIX filter(filterSize, VECTOR(filterSize)); // The filter to apply
	MatrixOperations::initFilter(filter);

	// load images
	string imageName("../images/blanc.png"); //Default filename
    if( argc > 1) {
        imageName = argv[1];
    }

	int width, height, bitDepth;
	unsigned char* image = stbi_load(imageName.c_str(), &width, &height, &bitDepth, 3);

    // Check for invalid input
    if( image == NULL ) {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }
	//Separate the channels
	int len = width * height;
	unsigned char red[len], green[len], blue[len];
	for(int i=0, j=0; i<3*len; i+=3, j++){
		red[j]   = image[i];
		green[j] = image[i+1];
		blue[j]  = image[i+2];
	}


	// Images images(clp.loadImages());

	/*
	for (auto image : images.getImages()) {
		// call kernel
		// show image
	}
	*/

	//Write the image to disk appending "_filter" to its name
	imageName = imageName.substr(0, imageName.length()-4);
	imageName += "_filter.png";
	stbi_write_png(imageName.c_str(), width, height, bitDepth, image, width*3);

	return 0;
}
