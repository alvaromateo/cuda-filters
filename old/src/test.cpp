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

#include "test.h"

/**
 * CommandLineParserTest methods
 */

CommandLineParserTest::CommandLineParserTest(const CommandLineParser *parser) : clp(parser) {}

void CommandLineParserTest::doTest() {
	printImages();
	printOptions();
}

void CommandLineParserTest::printImages() {
	std::cout << "Images: " << std::endl;
	std::vector<std::string>::const_iterator it = clp->getImages().begin();
	for ( ; it != clp->getImages().end(); ++it) {
		std::cout << *it << std::endl;
	}
}

void CommandLineParserTest::printOptions() {
	std::cout << "Options: " << std::endl;
	std::map<std::string, unsigned short>::const_iterator it = clp->opts.begin();
	while (it != clp->opts.end()) {
		std::cout << it->first << ": " << it->second << std::endl;
		++it;
	}
}