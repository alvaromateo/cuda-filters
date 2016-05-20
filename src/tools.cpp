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

// includes
#include "tools.h"


/**
 * Constructor for the parser. When creating the parser object this reads the command line
 * options and initializes the images vector and the options map.
 */
CommandLineParser::CommandLineParser(int argc, char **argv) {

}

/**
 * return: the filter size specified in the command line arguments or the default one
 */
unsigned int CommandLineParser::getFilterSize() {
	unsigned int size = DEFAULT_FILTER_SIZE;
	auto it = opts.find("size");
	if (it != opts.end()) {
		try {
			size = std::stoi(it->second);
		} catch (exception &e) {
			doHelp();
		}
	}
	return size;
}