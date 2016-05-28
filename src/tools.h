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
#include <cstdio>
#include <utility>
#include <sstream>


#define DEFAULT_FILTER_SIZE 5
#define DEFAULT_FILTER_TYPE 0

typedef std::vector<float> VECTOR;
typedef std::vector< VECTOR > MATRIX;

enum FilterType {
	blur,
	sharpen
};

/**
 * Parser for the command line options
 */
class CommandLineParser {
	friend class CommandLineParserTest;

	public:
		CommandLineParser(int &argc, char **&argv);
		inline const std::vector<std::string> &loadImages() const { return images; }
		unsigned short getFilterSize();

	private:
		std::vector<std::string> images;
		std::map<std::string, unsigned short> opts;

		void initOptions();
		std::string getOptionKey(const char *const &argument, int *index);
		unsigned short getOptionValue(const char *const &argument, const std::string &key);
		unsigned short transformTypeToInt(const std::string &type);
		bool isValid(std::string &key, int *index);
		bool isImage(const char *const &argument);
		void doHelp();
};

/**
 * Common operations for matrixes
 */
class MatrixOperations {
	public:
		static void initFilter(MATRIX &filter);
};

#endif
