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
#include <algorithm>


/*
 * Class Matrix that stores a 2D array of uchars. It is used to create the
 * filter and the different color frames of the images.
 */
template < typename T >
class Matrix {
	private:
		T *matrix;
		uint width;
		uint height;
		// this T is the one that we return when we subscript an index out
		// of bounds because of the filter position. Thanks to this we won't have
		// to take care of this problem.
		T trash;

		void swap(Matrix<T>& a, Matrix<T>& b) {
			using std::swap;
		    swap(a.matrix, b.matrix);
		    swap(a.width, b.width);
		    swap(a.height, b.height);
		    swap(a.trash, b.trash);
		}
		void copyMatrix(const T *matrix, T *mat) {
			for (uint i = 0; i < (this->width * this->height); ++i) {
				mat[i] = matrix[i];
			}
		}

	public:
		Matrix() : matrix(0), width(0), height(0), trash(0) {}
		Matrix(const T *matrix, uint w, uint h) : matrix(0), width(0), height(0), trash(0) {
			this->matrix = new T[width * height];
			copyMatrix(matrix, this->matrix);
		}
		Matrix(const Matrix<T> &matrix) : matrix(0), width(0), height(0), trash(0) {
			this->width = matrix.width;
			this->height = matrix.height;
		    this->trash = 0;
			this->matrix = new T[this->width * this->height];
			copyMatrix(matrix.matrix, this->matrix);
		}
		inline ~Matrix() { delete[] matrix; }
		inline Matrix<T> &operator=(Matrix<T> m) {
			swap(*this, m);
			return *this;
		}
		// Matrix &operator=(Matrix &&m);
		inline T &operator[](int index) {
			if (index < (width * height) && index >= 0) {
		        return this->matrix[index];
		    }
		    return trash;
		}
		// Matrix ops
		inline uint getWidth() const { return width; }
		inline uint getHeight() const { return height; }
		inline T *getMatrix() const { return matrix; }
		void setMatrix(const T *matrix) { copyMatrix(matrix, this->matrix); }

};


/*
 * Class that stores the RGB color frames of an image. Used to store the image returned by the
 * external library given the parameter of the file name.
 */
class Image {
	private:
		std::vector<Matrix<uchar> > img;
		int bitDepth;
		int width;
		int height;

		// bool greyscale;
		std::string imageName;
		std::vector<Matrix<uchar> > loadImageFromDisk(const std::string &imageName);

	public:
		Image() {}
		Image(const std::string &imageName); // Throws exception std::invalid_argument
		Image(const Image &otherImage);
		Matrix<uchar> &operator[](uint index);
		// Getters and setters
		inline std::vector<Matrix<uchar> > &getImg() { return img; }
		inline int getWidth() { return width; }
		inline int getHeight() { return height; }
		void setImage(const std::string &imageName); // Throws exception std::invalid_argument
		void saveImageToDisk();

};

#endif