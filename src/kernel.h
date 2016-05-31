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


#ifndef KERNEL
#define KERNEL

// Includes
#include "tools.h"
#include "image.h"


class Kernel {
	private:
		// private execution type variables
		ExecutionType executionType;
		Color color;
		unsigned short nThreads;
		bool pinned;
		unsigned int filterSize;

		// private kernel methods
		void sequentialExec(const uchar *filter, uchar *image, unsigned int imageSize);
		void singleCardSynExec(const uchar *filter, uchar *image, unsigned int imageSize);
		void singleCardAsynExec(const uchar *filter, uchar *image, unsigned int imageSize);
		void multiCardSynExec(const uchar *filter, uchar *image, unsigned int imageSize);
		void multiCardAsynExec(const uchar *filter, uchar *image, unsigned int imageSize);
		// new method with only one input-output matrix, overwriting it?
		
		// private methods for allocating memory
		void getPinnedMemory();
		void getMemory();
		
		// private methods to free memory
		void freePinnedMemory();
		void freeMemory();

	public:
		Kernel() : executionType(sequential), color(rgb), nThreads(THREADS), 
			pinned(false), filterSize(DEFAULT_FILTER_SIZE) {}
		Kernel(const CommandLineParser &clp);
		void applyFilter(const uchar *filter, uchar *image, unsigned int imageSize);
};


#endif