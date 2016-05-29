#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
	string imageName("blanc.png"); // by default
	string type("greyscale");
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
    int nCols, nRows;
    nCols = image.cols;
    nRows = image.rows;
    

    if(nChannels == 3){ //rgb
        /*for(int i=0; i< nRows; i++){
            for (int j=0; j<nCols; j++){
                image.at<Vec3b>(i,j) [0]= 128;
                image.at<Vec3b>(i,j) [1]= 128;
                image.at<Vec3b>(i,j) [2]= 128;
            }
        }*/
        //vector<Vec3b> vec(nRows*nCols);
        Mat bgr[3];
        split(image,bgr);
        vector<int> red, green, blue;
        //TODO assignar a tres posicions distintes
        if (image.isContinuous()) {
            blue.assign(bgr[0].datastart, bgr[0].dataend);
            green.assign(bgr[1].datastart, bgr[1].dataend);
            red.assign(bgr[2].datastart, bgr[2].dataend);
        }
        else {
            for (int i = 0; i < image.rows; ++i) {
                blue.insert(blue.end(), bgr[0].ptr<int>(i), bgr[0].ptr<int>(i)+bgr[0].cols);
                green.insert(green.end(), bgr[1].ptr<int>(i), bgr[1].ptr<int>(i)+bgr[1].cols);
                red.insert(red.end(), bgr[2].ptr<int>(i), bgr[2].ptr<int>(i)+bgr[2].cols);
            }
        }
    }
    else if(nChannels == 1){ //grayscale
        /*for(int i=0; i< nRows; i++){
            for (int j=0; j<nCols; j++){
                image.at<uchar>(i,j) = 128;
            }
        }*/
        //vector<uchar> vec(nRows*nCols);
        vector<int> vec;
        if (image.isContinuous()) {
            vec.assign(image.datastart, image.dataend);
        }
        else {
            for (int i = 0; i < image.rows; ++i) {
                vec.insert(vec.end(), image.ptr<uchar>(i), image.ptr<uchar>(i)+image.cols);
            }
        }
        
        //int *array = &vec[0]; //Opció 1

        int array[nRows*nCols]; //Opció 2
        copy(vec.begin(), vec.end(), array);

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