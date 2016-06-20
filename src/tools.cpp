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

#include <stdexcept>
#include <climits>
#include <cerrno>


/** 
 * Personal strtoul function, as the std::stoul is only supported from C++11 and nvcc
 * is using C++98. Converts an string to a number.
 */
unsigned short my_stoul (const char *str, size_t *idx = 0, int base = 10) {
    char *endp;
    unsigned short value = strtoul(str, &endp, base);
    if (endp == str) {
        throw std::invalid_argument("my_stoul");
    }
    if (value == ULONG_MAX && errno == ERANGE) {
        throw std::out_of_range("my_stoul");
    }
    if (idx) {
        *idx = endp - str;
    }
    return value;
}

/**
 * Function to check if a number is a power of 2 or not.
 */
inline bool is_power_of_2(const unsigned short &n) {
	return (n & (n - 1)) == 0;
}


/**
 * CommandLineParser public methods
 */


/**
 * Constructor for the parser. When creating the parser object this reads the command line
 * options and initializes the images vector and the options map.
 */
CommandLineParser::CommandLineParser(int &argc, char **&argv) {
	int i = 1; // first argv parameter is the program name
	bool allImagesFound = false;
	while (!allImagesFound && i < argc) {
		if (isImage(argv[i])) {
			images.push_back(std::string(argv[i]));
		} else {
			allImagesFound = true; // exit the loop
			// we substract one because right now i points to the next element to process
			// and after the else we add 1 to it.
			--i;
		}
		++i;
	}

	// user must input at least one image
	if (images.empty()) {
		doHelp(); // exit the program showing the usage
	}
	// init map with default values
	initOptions();
	while (i < argc) {
		std::string key = getOptionKey(argv[i], &i);
		if (i < argc) {
			std::map<std::string, unsigned short>::iterator it = opts.find(key);
			it->second = getOptionValue(argv[i], key);
		} else {
			doHelp();
		}
		++i;
	}
}


/**
 * CommandLineParser private methods
 */


/**
 * This method initializes the options variable with the keys and values of
 * the default options.
 */
void CommandLineParser::initOptions() {
	opts.insert(std::pair<std::string, unsigned short> (std::string("filter"), DEFAULT_FILTER_TYPE));
	opts.insert(std::pair<std::string, unsigned short> (std::string("show"), 0));
	opts.insert(std::pair<std::string, unsigned short> (std::string("threads"), THREADS));
	opts.insert(std::pair<std::string, unsigned short> (std::string("exec"), sequential));
	opts.insert(std::pair<std::string, unsigned short> (std::string("pinned"), 0));
	opts.insert(std::pair<std::string, unsigned short> (std::string("color"), rgb));
}

/**
 * This method receives an argv[i] parameter and checks it is a valid option
 * for the program. If it is then the parameter is transformed to a string.
 *
 * return: the string containing the user command line option.
 */
std::string CommandLineParser::getOptionKey(const char *const &argument, int *index) {
	std::string key;
	if (argument[0] == '-') {
		if (argument[1] == '-') {
			key = std::string(argument + 2);
		} else {
			key = std::string(argument + 1);
		}
	}
	if (!isValid(key, index)) {
		doHelp(); // option not valid. Show usage and exit
	}
	return key;
}

/**
 * This method receives an argv[i] parameter that contains a value for an option
 * previously read and the key string corresponding to this value and transforms
 * the argv[i] value into a number to fit in the opts map.
 *
 * return: the number containing the value for the given option.
 */
unsigned short CommandLineParser::getOptionValue(const char *const &argument, const std::string &key) {
	unsigned short value = 0;
	if (key == "show" || key == "pinned") {
		value = 1;
	} else if (key == "threads") {
		try {
			value = my_stoul(argument);
		} catch (std::exception &e) {
			doHelp();
		}
		if ((value > MAX_THREAD_NUMBER) && !is_power_of_2(value)) {
			doHelp();
		}
	} else if (key == "filter" || key == "color" || key == "exec") {
		// transform the string into a number with transformTypeToInt
		value = transformTypeToInt(std::string(argument));
	} else {
		doHelp();
	}
	return value;
}

/**
 * This method transforms a filter type string into an unsigned short so that it can
 * be stored in the options map.
 *
 * return: the number corresponding to the string filter type.
 */
unsigned short CommandLineParser::transformTypeToInt(const std::string &type) {
	unsigned short typeNum = 0;
	// Add different filter types here and in the tools.h enum
	if (!type.compare("avg3")) {
		typeNum = avg3;
	} else if (!type.compare("avg5")) {
		typeNum = avg5;
	} else if (!type.compare("sharpenWeak")) {
		typeNum = sharpenWeak;
	} else if (!type.compare("sharpenStrong")) {
		typeNum = sharpenStrong;
	} else if (!type.compare("gaussian3")) {
		typeNum = gaussian3;
	} else if (!type.compare("gaussian5")) {
		typeNum = gaussian5;
	} else if (!type.compare("edgeDetection")) {
		typeNum = edgeDetection;
	} else if (!type.compare("embossing")) {
		typeNum = embossing;
	} else if (!type.compare("rgb")) {
		typeNum = rgb;
	} else if (!type.compare("grayscale")) {
		typeNum = grayscale;
	} else if (!type.compare("sequential")) {
		typeNum = sequential;
	} else if (!type.compare("singleCardSyn")) {
		typeNum = singleCardSyn;
	} else if (!type.compare("singleCardAsyn")) {
		typeNum = singleCardAsyn;
	} else if (!type.compare("multiCardSyn")) {
		typeNum = multiCardSyn;
	} else if (!type.compare("multiCardAsyn")) {
		typeNum = multiCardAsyn;
	} else {
		doHelp(); // type not valid. Show usage and exit the program
	}
	return typeNum;
}

/*
 * This method contains all the valid options that can be passed through command
 * line arguments. If the option must have a value given as input then the index
 * pointing to "argv" is incremented. If it doesn't need a value (it's just a true
 * if set, false otherwise option) then the index is not incremented.
 *
 * return: a boolean indicating if the option read is valid or not.
 */
bool CommandLineParser::isValid(std::string &key, int *index) {
	bool valid = false;
	if (!key.empty()) {
		if (key == "filter" || key == "threads" || key == "exec" || key == "color") {
			valid = true;
			++(*index); // increment index to check for the value
		} else if (key == "show") {
			// just set the option as valid and don't increment index
			// because this option doesn't have any value
			valid = true; 
		} else if (key == "s") {
			// set valid to true and change key to show to avoid saving "show" and
			// "s" options, which are the same
			key = "show";
			valid = true;
		} else if (key == "pinned") {
			valid = true;
		} else if (key == "p") {
			key = "pinned";
			valid = true;
		}
	}
	return valid;
}

/*
 * This method recieves a name of an image file and detects if it is a valid image
 * by looking at its extension (.png, .jpg, etc). If it doesn't have an extension included
 * in the supported images formats then it returns false.
 *
 * return: a boolean indicating if the name of the image has a valid format or not
 */
bool CommandLineParser::isImage(const char *const &argument) {
	std::string image(argument);
	std::string::size_type pos = image.find_last_of('.');
	if (!image.compare(pos+1, 3, "png")) {
		return true;
	}
	return false;
}

/*
 * This method prints the usage of the program.
 */
void CommandLineParser::doHelp() {
	std::ostringstream help;
	help << "Usage: cudafilters.exe image.png [image2.png image3.png ...] options" << std::endl;
	help << "Options can be:" << std::endl;
	help << "	--filter f 	where f is one of the following filter types:" << std::endl;
	help << "					avg3 (default), avg5, sharpenWeak, sharpenStrong, gaussian3, gaussian5, edgeDetection, embossing" << std::endl;
	// help << "	--show|-s	if set, the modified images are opened when the program finishes" << std::endl;
	help << "	--pinned|-p 	if set, the program will use pinned memory" << std::endl;
	help << "	--exec e 	where e is one of the following execution types:" << std::endl;
	help << "					sequential, singleCardSyn, singleCardAsyn, multiCardSyn, multiCardAsyn" << std::endl;
	// color option not working in this version
	help << "	--threads t where t is an integer number power of 2 and not greater than " << MAX_THREAD_NUMBER <<
		" specifying the number of threads in each dimension" << std::endl;
	help << "Pinned memory is mandatory in case of asyncronous execution" << std::endl;
	help << "Currently supported images formats: .png";
	help.flush();
	std::cout << help.str() << std::endl;
	std::exit(1);
}

