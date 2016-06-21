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
#include "readCommandLine.h"


/*
 * This method prints the usage of the program.
 */
void doHelp() {
	printf("Usage: filters.exe image.png [image2.png image3.png ...] options\n");
	printf("Options can be:\n");
	printf("	--filter f 		where f is one of the following filter types:\n");
	printf("						avg3 (default), avg5, sharpenWeak, sharpenStrong, gaussian3, gaussian5, edgeDetection, embossing\n");
	printf("	--pinned		if set, the program will use pinned memory\n");
	printf("	--threads t 	where t is an integer number power of 2 and not greater than %u\n", MAX_THREAD_NUMBER);
	printf("Pinned memory is mandatory in case of asyncronous execution\n");
	printf("Currently supported images formats: .png\n");
	exit(1);
}


/*
 * This method recieves a name of an image file and detects if it is a valid image
 * by looking at its extension (.png, .jpg, etc). If it doesn't have an extension included
 * in the supported images formats then it returns false.
 *
 * return: a boolean indicating if the name of the image has a valid format or not
 */
bool isImage(const char *const &argument) {
	std::string::size_type pos = image.find_last_of('.');
	if (!image.compare(pos+1, 3, "png")) {
		return true;
	}
	return false;
}