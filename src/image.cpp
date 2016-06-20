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
 * Image public methods
 */


 /*
  * Constructor to create an image given a name that corresponds to an image in
  * the file system.
  */
Image::Image(const std::string &imageName) : width(0), height(0), bitDepth(0) {
    this->imageName = imageName;
	this->img = loadImageFromDisk(imageName);
#if DEBUG
    std::cerr << "Image::Image width: " << width << std::endl;
#endif
}

/*
 * Copy constructor
 */
Image::Image(const Image &otherImage) : width(otherImage.width), height(otherImage.height), bitDepth(otherImage.bitDepth) {
    this->imageName = otherImage.imageName;
    this->img = std::vector<Matrix<uchar> > (3);
    for (int i = 0; i < otherImage.img.size(); ++i) {
        Matrix<uchar> mat(otherImage.img[i]);
        img[i] = mat;
    }
}

/*
 * Redefinition of the subscript operator to allow access to the different color
 * frames inside this object. index can be one of 3 values.
 *      red = 0
 *      green = 1
 *      blue = 2
 *
 * return: returns a Matrix object corresponding to the channel color provided by
 * index.
 */
Matrix<uchar> &Image::operator[](uint index) {
    if (index < 3) {
        return this->img[index];
    }
    // if someone is trying to access an invalid element we return the red color
    // channel
    return trash;
}

/*
 * Set a new image for this object from a file named imageName.
 */
void Image::setImage(const std::string &imageName) {
    this->imageName = imageName;
	this->img = loadImageFromDisk(imageName);
}

/*
 * Private function to load an image with the stbi library. If the file name provided
 * doesn't correspond to any image, then an std::invalid_argument exception is thrown.
 * 
 * return: a vector with 3 matrix objects, one for each color channel of an image.
 */
std::vector<Matrix<uchar> > Image::loadImageFromDisk(const std::string &imageName) {
    uchar* image = stbi_load(imageName.c_str(), &width, &height, &bitDepth, 3);

    // Check for invalid input
    if (image == NULL) {
    	throw std::invalid_argument("Could not open or find the image");
    }

    //Separate the channels
    int len = width * height;
    uchar red[len], green[len], blue[len];
    for (int i = 0, j = 0; i < 3*len; i += 3, ++j){
        red[j]   = image[i];
        green[j] = image[i+1];
        blue[j]  = image[i+2];
    }

    std::vector<Matrix<uchar> > img;
    Matrix<uchar> redMat(red, width, height);
    Matrix<uchar> greenMat(green, width, height);
    Matrix<uchar> blueMat(blue, width, height);
    img.push_back(redMat);
    img.push_back(greenMat);
    img.push_back(blueMat);
    std::free(image);

#if DEBUG
    std::cerr << "Image::loadImageFromDisk width: " << width << std::endl;
#endif

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
    int len = width * height;
#if DEBUG
    std::cerr << "Image::saveImageToDisk. len: " << len << std::endl;
#endif
    uchar *image = new uchar[len*3];
    for (int i = 0, j = 0; i < 3*len; i += 3, ++j) {
        image[i] = img[0][j];
        image[i+1] = img[1][j];
        image[i+2] = img[2][j];
    }
    stbi_write_png(imageName.c_str(), width, height, bitDepth, &image[0], width*3);
    delete[] image;
}
