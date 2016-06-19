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


#ifndef IMAGE
#define IMAGE

// Includes
#include "tools.h"


/*
 * Class Matrix that stores a 2D array of uchars. It is used to create the
 * filter and the different color frames of the images.
 */
class Matrix {
	private:
		uchar *matrix;
		uint width;
		uint height;
		// this uchar is the one that we return when we subscript an index out
		// of bounds because of the filter position. Thanks to this we won't have
		// to take care of this problem.
		uchar trash;

		void copyMatrix(const uchar *matrix, uchar *mat);

	public:
		Matrix() : matrix(0), width(0), height(0), trash(0) {}
		Matrix(const uchar *matrix, uint w, uint h);
		Matrix(const Matrix &matrix);
		~Matrix();
		uchar &operator[](int index);
		// Matrix ops
		inline uint getWidth() const { return width; }
		inline uint getHeight() const { return height; }
		inline uchar *getMatrix() const { return matrix; }
		void setMatrix(const uchar *matrix);
};


// Filters available
float filter_avg3[9] = {1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9};
float filter_avg5[25] = {1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25, 1./25};
float filter_sharpenWeak[9] = {0,-1,0,-1,5,-1,0,-1,0};
float filter_sharpenStrong[9] = {-1,-1,-1,-1,9,-1,-1,-1,-1};
float filter_gaussian3[9] = {1./16, 2./16, 1./16, 2./16, 4./16, 2./16, 1./16, 2./16, 1./16};
float filter_gaussian5[25] = {1./256, 4./256, 6./256, 4./256, 1./256, 4./256, 16./256, 24./256, 16./256, 4./256, 6./256, 24./256, 36./256, 24./256, 6./256, 4./256, 16./256, 24./256, 16./256, 4./256, 1./256, 4./256, 6./256, 4./256, 1./256};
float filter_edgeDetection[9] = {0,1,0,1,-4,1,0,1,0}; //Normalize result by adding 128 to all elements
float filter_embossing[9] = {-2,-1,0,-1,1,1,0,1,2};

class Filter {
	private:
		float *filter;
		uint size;
		float trash;

	public:
		Filter() {}
		Filter(uchar filterType);
		// Filter ops
		float &operator[](int index);
		inline float *getFilter() const { return filter; }
		inline uint getSize() const { return size; }
};


/*
 * Class that stores the RGB color frames of an image. Used to store the image returned by the
 * external library given the parameter of the file name.
 */
class Image {
	private:
		std::vector<Matrix> img;
		int bitDepth;
		int width;
		int height;

		// bool greyscale;
		std::string imageName;
		std::vector<Matrix> loadImageFromDisk(const std::string &imageName);

	public:
		Image() : img() {}
		Image(const std::string &imageName); // Throws exception std::invalid_argument
		Image(const Image &otherImage);
		Matrix &operator[](uint index);
		// Getters and setters
		inline std::vector<Matrix> &getImg() { return img; }
		void setImage(const std::string &imageName); // Throws exception std::invalid_argument
		void saveImageToDisk();

};

#endif