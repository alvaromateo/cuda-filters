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


static char *defaultImage = "images/lena.png";


/*
 * This method prints the usage of the program.
 */
void doHelp() {
	printf("Usage: filters.exe image.png options\n");
	printf("Options can be:\n");
	printf("	-f filter		where filter is one of the following filter types:\n");
	printf("						0 	avg3 (default)\n");
	printf("						1 	avg5\n");
	printf("						2 	sharpenWeak\n");
	printf("						3 	sharpenStrong\n");
	printf("						4 	gaussian3\n");
	printf("						5 	gaussian5\n");
	printf("						6 	edgeDetection\n");
	printf("						7 	embossing\n");
	printf("	-p				if set, the program will use pinned memory\n");
	printf("	-t threads 		where t is an integer number not greater than %u\n", MAX_THREADS);
	printf("Pinned memory is mandatory in case of asyncronous execution\n");
	printf("Currently supported images formats: .png\n");
	exit(1);
}


/*
 * This method recieves a name of an image file and detects if it is a valid image
 * by looking at its extension (.png, .jpg, etc). If it doesn't have an extension included
 * in the supported images formats then it returns false.
 *
 * return: the char array with the valid image or NULL if the argument wasn't a valid one
 */
char *getValidImage(char *argument) {
	char *lastOccurrence = strstr(argument, ".png");
	if (lastOccurrence) {
		return argument;
	}
	return NULL;
}

/*
 * This method passes argc and argv and extracts in "filter, threads and pinned" the
 * values given to the options of the program. In the return value there's the image
 * name.
 *
 * return: the image name or "lena.png" if no parameters given
 */
char *getOptions(int argc, char **argv, uchar *filter, uchar *threads, uchar *pinned) {
	char *imageName;
	// Default values
	imageName = defaultImage;
	*filter = FILTER;
	*threads = THREADS;
	*pinned = PINNED;

	if (argc > 1) {
		uchar i = 1; // first argv parameter is the program name
		if (imageName = getValidImage(argv[i])) {
			++i;
		} 

		while (i < argc) {
			char key = getOptionKey(argv[i]);
			switch (key) {
				case 'f':
					++i; // add one to fetch the following value
					if (i == argc) doHelp(); // if there is no parameter value we exit
					*filter = atoi(argv[i]);
					break;
				case 'p':
					*pinned = 1;
					break;
				case 't':
					++i; // add one to fetch the following value
					if (i == argc) doHelp(); // if there is no parameter value we exit
					*filter = atoi(argv[i]);
					if (*filter > MAX_THREADS) doHelp();
					break;
			}
			++i;
		}
	}

	return imageName;
}


/*
 * This method receives an argv[i] parameter and checks it is a valid option
 * for the program. If it is then the parameter is transformed to a char.
 *
 * return: the char containing the user command line option
 */
char getOptionKey(const char *argument) {
	if (strlen(argument) == 2) {
		if (argument[0] == '-') {
			if (isValidKey(argument[1])) {
				return argument[1];
			}
		}
	}
	// If the above conditions are not true print help and exit
	doHelp();
}

/*
 * This method checks that the key is valid (is one of 'f', 'p' or 't').
 *
 * return: 0 if key no valid and 1 if key valid
 */
uchar isValidKey(char key) {
	switch (key) {
		case 'f':
		case 'p':
		case 't':
			return 1;
	}
	return 0;
}