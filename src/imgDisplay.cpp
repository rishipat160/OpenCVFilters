/**
 * Rishi Patel
 * Due: 01/23/2025
 * 
 * This purpose of this file is to display an image.
 * The image display code template 
 * 
 */

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/**
 * main
 * 
 * This function displays an image.
 * 
 * Input:
 *      int argc - the number of arguments
 *      char** argv - the arguments
 * Output:
 *      int - 0 if successful, -1 if not
 */
int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: " << argv[0] << " ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], IMREAD_COLOR); // Read the file

    if( image.empty() ) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl ;
        return -1;
    }

    namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    imshow( "Display window", image ); // Show our image inside it.

    while(waitKey(10) != 'q');
    return 0;
}
