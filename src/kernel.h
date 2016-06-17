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


/*
 * Class to contain all the variables needed to run the kernel.
 * Call to applyFilter will use the kernel according to the options provided
 * by the command-line.
 */
class Kernel {
	private:
		// private execution type variables
		ExecutionType executionType;
		Color color;
		unsigned short nThreads;
		bool pinned;

		// private kernel methods
		void sequentialExec(const Filter &filter, Image &image);
		void singleCardSynExec(const Filter &filter, Image &image);
		void singleCardAsynExec(const Filter &filter, Image &image);
		void multiCardSynExec(const Filter &filter, Image &image);
		void multiCardAsynExec(const Filter &filter, Image &image);
		// private methods for allocating memory
		void getPinnedMemory();
		void getMemory();
		// private methods to free memory
		void freePinnedMemory();
		void freeMemory();

	public:
		Kernel() : executionType(sequential), color(rgb), nThreads(THREADS), pinned(false) {}
		Kernel(const CommandLineParser &clp);
		static void applyFilter(const Filter &filter, Image &image);
};


#endif