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

#ifndef TOOLS
#define TOOLS

// includes
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <exception>

#define DEFAULT_FILTER_SIZE 3
#define DEFAULT_FILTER_TYPE 0

typedef std::vector<float> VECTOR;
typedef std::vector< VECTOR > MATRIX;

/**
 * This enums are for the filter types and for the different command line options.
 * If a new option or type is implemented, add it to the corresponding enum and also
 * add the corresponding switch case.
 */
 
enum FilterTypes { 
	blur = "blur", 
	sharpen = "sharpen" 
};

enum CommandLineOptions {
	type 
};

/**
 * Parser for the command line options
 */
class CommandLineParser {
	public:
		CommandLineParser(int argc, char **argv);
		std::vector<string> &getImages() { return images; }
		unsigned int getFilterSize();

	private:
		std::vector<string> images;
		std::map<string, string> opts;

		void doHelp();
		bool analyze(int argc, char **argv);
};

/**
 * Common operations for matrixes
 */
class MatrixOperations {
	public:
		static void initFilter(FILTER &filter);
};

#endif
