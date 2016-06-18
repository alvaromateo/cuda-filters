#include <iostream>
#include <string>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;

int main(int argc, char **argv) {
	string imageName("blanc.png"); // by default
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

	//Write the image to disk appending "_filter" to its name
	imageName = imageName.substr(0, imageName.length()-4);
	imageName += "_filter.png";
	stbi_write_png(imageName.c_str(), width, height, bitDepth, image, width*3);

    return 0;
}
