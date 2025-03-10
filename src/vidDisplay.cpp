/**
 * Rishi Patel
 * Due: 01/23/2025
 * 
 * This is the main file for the video display program.
 * This program is a video display that allows the user to toggle filters on and off
 * and save frames to a file. There are many filters to choose from, including grayscale,
 * sepia, blur, sobel, blur quantization, face detection, cartoon, sketch, and more.
 * 
 * The program uses OpenCV to capture video from the camera and apply the filters.
 * It also uses the faceDetect.h and filter.h files to detect faces and apply filters.
 */

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include "faceDetect.h"
#include "filter.h"

/**
 * Main function for the video display program.
 * This function initializes the camera, applies the filters, and displays the video.
 * 
 * @param argc The number of arguments passed to the program.
 * @param argv The arguments passed to the program.
 * @return The exit status of the program.
 */
int main(int argc, char *argv[]) {
        cv::VideoCapture *capdev;
        printf("Opening video device\n");
        capdev = new cv::VideoCapture(0 + cv::CAP_DSHOW);  

        if(!capdev->isOpened()) {
                printf("Unable to open video device\n");
                delete capdev;
                return(-1);
        }
        printf("Opened\n");
        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Video", 1); // identifies a window
        cv::Mat frame;
        std::vector<cv::Rect> faces;
        
        enum FilterType { 
            GRAYSCALE, ALT_GRAY, ALT_GRAY2, SEPIA, BLUR1, BLUR2, 
            FACES, SOBEL_X, SOBEL_Y, MAGNITUDE, BLUR_QUANT, 
            CARTOON, SKETCH
        };
        bool filters[13] = {false};
        int quantizeLevels = 10;

        for(;;) {
                *capdev >> frame; // get a new frame from the camera, treat as a stream
                if( frame.empty() ) {
                  printf("frame is empty\n");
                  break;
                }

                // Filter application          
                if(filters[GRAYSCALE]) cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
                if(filters[ALT_GRAY]) alternativeGrayscale(frame, frame);
                if(filters[SEPIA]) sepia_filter(frame, frame);
                if(filters[BLUR1]) blur5x5_1(frame, frame);
                if(filters[BLUR2]) blur5x5_2(frame, frame);
                if(filters[MAGNITUDE]) {
                    cv::Mat sobelX_output, sobelY_output;
                    sobelX3x3(frame, sobelX_output);
                    sobelY3x3(frame, sobelY_output);
                    magnitude(sobelX_output, sobelY_output, frame);
                }
                if(filters[SOBEL_X]) {
                    cv::Mat sobel_output;
                    sobelX3x3(frame, sobel_output);
                    cv::convertScaleAbs(sobel_output, frame, 2.0);
                }
                if(filters[SOBEL_Y]) {
                    cv::Mat sobel_output;
                    sobelY3x3(frame, sobel_output);
                    cv::convertScaleAbs(sobel_output, frame, 2.0);
                }
                if(filters[BLUR_QUANT]) blurQuantize(frame, frame, quantizeLevels);
                if(filters[FACES]) {
                    cv::Mat grey;
                    cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
                    detectFaces(grey, faces);
                    drawBoxes(frame, faces);
                }
                if(filters[CARTOON]) cartoon_filter(frame, frame);
                if(filters[SKETCH]) sketch_filter(frame, frame);
                if(filters[ALT_GRAY2]) alternativeGrayscale3(frame, frame);

                cv::imshow("Video", frame);

                char key = cv::waitKey(10);
                switch(key) {
                    case 'q': goto cleanup;  // quit
                    case 's': {  // save frame
                        static int count = 0;
                        cv::imwrite("image_" + std::to_string(count++) + ".jpg", frame);
                        break;
                    } // Filter toggle
                    case 'g': filters[GRAYSCALE] = !filters[GRAYSCALE]; break;
                    case 'h': filters[ALT_GRAY] = !filters[ALT_GRAY]; break;
                    case 'e': filters[SEPIA] = !filters[SEPIA]; break;
                    case 'b': filters[BLUR1] = !filters[BLUR1]; break;
                    case 'n': filters[BLUR2] = !filters[BLUR2]; break;
                    case 'x': filters[SOBEL_X] = !filters[SOBEL_X]; break;
                    case 'y': filters[SOBEL_Y] = !filters[SOBEL_Y]; break;
                    case 'm': 
                        filters[MAGNITUDE] = !filters[MAGNITUDE];
                        filters[SOBEL_X] = filters[SOBEL_Y] = false;
                        break;
                    case 'i': filters[BLUR_QUANT] = !filters[BLUR_QUANT]; break;
                    case 'f': filters[FACES] = !filters[FACES]; break;
                    case 'c': filters[CARTOON] = !filters[CARTOON]; break;
                    case 'k': filters[SKETCH] = !filters[SKETCH]; break;
                    case 'j': filters[ALT_GRAY2] = !filters[ALT_GRAY2]; break;
                }
        }

cleanup:
        // Clean up
        if(capdev != nullptr) {
            capdev->release();  // Release the camera
            delete capdev;      // Free the memory
            capdev = nullptr;
        }
        cv::destroyAllWindows();  // Close any OpenCV windows
        return(0);
}