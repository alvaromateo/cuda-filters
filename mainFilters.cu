include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
	string imageName("blanc.png"); // by default
	string type;
    if( argc > 1) {
        imageName = argv[1];
        type = argv[2];
    }

    Mat image;
    if(type == "rgb"){
    	image = imread(imageName.c_str(), IMREAD_COLOR); // Read the file
    }
    else if(type == "grayscale" || type == "greyscale"){
    	image = imread(imageName.c_str(), IMREAD_GRAYSCALE); // Read the file
    }

    //Convert to grey. TODO filter

    int nChannels = image.channels();

    if(nChannels == 3){ //rgb
        for(int i=0; i< r; i++){
            for (int j=0; j<c; j++){
                image.at<Vec3b>(i,j) [0]= 128;
                image.at<Vec3b>(i,j) [1]= 128;
                image.at<Vec3b>(i,j) [2]= 128;
            }
        }
    }
    else if(nChannels == 1){ //grayscale
        for(int i=0; i< r; i++){
            for (int j=0; j<c; j++){
                image.at<uchar>(i,j) = 128;
            }
        }
    }
    
    
    

    if( image.empty() ) {                     // Check for invalid input
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    //imwrite("output.png", image); //Write to disk


    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", image );                // Show our image inside it.

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}