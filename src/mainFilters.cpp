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
#include "tools.h"
#include "test.h"


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
	// Images images(clp.loadImages());

	/*
	for (auto image : images.getImages()) {
		// call kernel
		// show image
	}
	*/
}