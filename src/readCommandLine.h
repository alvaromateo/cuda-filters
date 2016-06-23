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

#ifndef READ_COMMAND_LINE
#define READ_COMMAND_LINE


// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Defines
#define NAME_SIZE 64
#define FILTER 0
#define PINNED 0
#define THREADS 8
#define MAX_THREADS 32


// Typedefs
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;


// Methods
void doHelp();
char *getValidImage(char *argument);
uchar isValidKey(char key);
char getOptionKey(const char *argument);

// Call getOptions  to fetch the image name and initialize the rest of parameters
char *getOptions(int argc, char **argv, uchar *filter, uchar *threads, uchar *pinned);



#endif