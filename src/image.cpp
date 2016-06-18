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
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


/**
 * Matrix public methods
 */


 /*
  * Constructor that creates a matrix stored in the heap with the given width
  * and height.
  */
Matrix::Matrix(const uchar *matrix, uint w, uint h) : matrix(nullptr), width(w), height(h) {
	this.matrix = new uchar[width * height];
	copyMatrix(matrix, this.matrix);
}

/*
 * Copy constructor.
 */
Matrix::Matrix(const Matrix &matrix) {
	this.width = matrix.width;
	this.height = matrix.height;
	this.matrix = new uchar[this.width * this.height];
	copyMatrix(matrix.matrix, this.matrix);
}

/*
 * Destructor to free memory.
 */
Matrix::~Matrix() {
	delete [] this.matrix;
}

/*
 * Function that given a filter type initializes this.matrix with the values
 * required for that filter to work inside the kernel.
 */
void Matrix::initializeFilter(uchar filterType) {
	// TODO: implement the filter initializer
}


/*
 * Matrix private methods
 */


/*
 * Method that copies an array.
 */
void Matrix::copyMatrix(const uchar *matrix, uchar &mat) {
	for (uint i = 0; i < (this.width * this.height); ++i) {
		mat[i] = matrix[i];
	}
}


/**
 * Image public methods
 */


 /*
  * Constructor to create an image given a name that corresponds to an image in
  * the file system.
  */
Image::Image(const std::string &imageName) {
    this.imageName = imageName;
    try {
    	this.img = loadImageFromdisk(imageName, this.width, this.height);
    } catch (const std::invalid_argument &e) {
    	// Re-throw exception
    	throw e;
    }
}

/*
 * Set a new image for this object from a file named imageName.
 */
void Image::setImage(const std::string &imageName) {
    this.imageName = imageName;
	try {
		this.img = loadImageFromdisk(imageName, this.width, this.height);
	} catch (const std::invalid_argument &e) {
		// Re-throw exception
		throw e;
	}
}

/*
 * Private function to load an image with the stbi library. If the file name provided
 * doesn't correspond to any image, then an std::invalid_argument exception is thrown.
 * 
 * return: a vector with 3 matrix objects, one for each color channel of an image.
 */
std::vector<Matrix> Image::loadImageFromDisk(const std::string &image, uint &width, uint &height) {
	int bitDepth;
    unsigned char* image = stbi_load(imageName.c_str(), &width, &height, &bitDepth, 3);

    // Check for invalid input
    if (image == NULL) {
    	throw std::invalid_argument("Could not open or find the image");
    }

    //Separate the channels
    int len = width * height;
    unsigned char red[len], green[len], blue[len];
    for (int i = 0, j = 0; i < 3*len; i += 3, ++j){
        red[j]   = image[i];
        green[j] = image[i+1];
        blue[j]  = image[i+2];
    }

    std::vector<Matrix> img(3);
    Matrix red(red, width, height);
    Matrix green(green, width, height);
    Matrix blue(blue, width, height);
    img[0] = red;
    img[1] = green;
    img[2] = blue;

    return img;
}

/*
 * Function that saves the current image to disk with the name:
 * [currentName]_filter.png
 * It always saves the images in .png format.
 */
void Image::saveImageToDisk() {
	//Write the image to disk appending "_filter" to its name
    imageName = imageName.substr(0, imageName.length()-4);
    imageName += "_filter.png";
    stbi_write_png(imageName.c_str(), width, height, bitDepth, image, width*3);
}
