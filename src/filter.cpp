/**
 * Rishi Patel
 * Due: 01/23/2025
 * 
 * This purpose of this file is to implement the filter functions.
 * The different filters are implemented in the following functions:
 * 
 * 1. alternativeGrayscale
 * 2. sepia_filter
 * 3. blur5x5_1
 * 4. blur5x5_2
 * 5. alternativeGrayscale3
 * 6. sobelX3x3
 * 7. sobelY3x3
 * 8. magnitude
 * 9. blurQuantize
 * 10. cartoon_filter
 * 11. sketch_filter
 * 
 */


#include "filter.h"



/**
 * alternativeGrayscale
 * 
 * This function implements an alternative grayscale filter.
 * It creates a grayscale image by averaging the blue and green channels,
 * ignoring the red channel. Each output pixel's RGB values are set to
 * this average value.
 *  * Input:
 *      cv::Mat &src - the source image
 *      cv::Mat &dst - the destination image
 * Output:
 *      int - 0 if successful, -1 if not
 */
int alternativeGrayscale(cv::Mat &src, cv::Mat &dst) {
    // Check if input image is empty
    if (src.empty()) {
        return -1;
    }

    // Create destination image of same size as source
    dst.create(src.size(), CV_8UC3);  // 8-bit, 3 channels 
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // Get pixel value
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            // Set pixel value
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(
                pixel[0] * 0.5 + pixel[1] * 0.5,
                pixel[0] * 0.5 + pixel[1] * 0.5,
                pixel[0] * 0.5 + pixel[1] * 0.5
            );
        }
    }
    return 0;
}

/**
 * sepia_filter
 * 
 * This function implements a sepia filter.
 * It uses the sepia matrix to convert the image to sepia.
 * Input:
 *      cv::Mat &src - the source image
 *      cv::Mat &dst - the destination image
 * Output:
 *      int - 0 if successful, -1 if not
 */
int sepia_filter(cv::Mat &src, cv::Mat &dst) {
    // Check if input image is empty
    if (src.empty()) {
        return -1;
    }

    // Create destination image of same size as source
    dst.create(src.size(), CV_8UC3);

    // For each pixel in the image
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // Get original BGR values (OpenCV uses BGR order)
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            double blue = pixel[0];
            double green = pixel[1];
            double red = pixel[2];

            // Calculate new values using original RGB values
            // Clamp values to 255 using std::min
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(
                std::min(255.0, blue * 0.131 + green * 0.534 + red * 0.272),  // Blue
                std::min(255.0, blue * 0.168 + green * 0.686 + red * 0.349),  // Green
                std::min(255.0, blue * 0.189 + green * 0.769 + red * 0.393)   // Red
            );
        }
    }
    return 0;
}

/**
 * blur5x5_1
 * 
 * This function implements a Gaussian blur5x5_1 filter.
 * Input:
 *      cv::Mat &src - the source image
 *      cv::Mat &dst - the destination image
 * Output:
 *      int - 0 if successful, -1 if not
 */
int blur5x5_1(cv::Mat &src, cv::Mat &dst) {
    // Check if input image is empty
    if (src.empty()) {
        return -1;
    }

    // Create destination image of same size as source
    dst.create(src.size(), CV_8UC3);  // 8-bit, 3 channels  

    int kernel[5][5] = {
                {1, 2, 4, 2, 1},
                {2, 4, 8, 4, 2},
                {4, 8, 16, 8, 4},
                {2, 4, 8, 4, 2},
                {1, 2, 4, 2, 1}
            };

    // For each pixel in the image
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // Get original BGR values (OpenCV uses BGR order)
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);

            // Copy src to dst first
            dst.at<cv::Vec3b>(i, j) = pixel;

            // Skip border pixels (first/last two rows and columns)
            if (i < 2 || i >= src.rows - 2 || j < 2 || j >= src.cols - 2)
                continue;

            // Variables to store sum of each channel
            int sumB = 0, sumG = 0, sumR = 0;

            // Process 5x5 kernel
            for (int k = -2; k <= 2; k++) {
                for (int l = -2; l <= 2; l++) {
                    // Get the neighboring pixel
                    cv::Vec3b neighborPixel = src.at<cv::Vec3b>(i + k, j + l);
                    
                    // Add weighted values to sums
                    sumB += neighborPixel[0] * kernel[k + 2][l + 2];
                    sumG += neighborPixel[1] * kernel[k + 2][l + 2];
                    sumR += neighborPixel[2] * kernel[k + 2][l + 2];
                }
            }

            // Normalize by dividing by sum of kernel weights (100)
            dst.at<cv::Vec3b>(i, j)[0] = cv::saturate_cast<uchar>(sumB / 100);
            dst.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(sumG / 100);
            dst.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(sumR / 100);
        }
    }

    return 0;
}

/**
 * blur5x5_2
 * 
 * This function implements a separable Gaussian blur filter using two 1D passes.
 * It applies a horizontal pass followed by a vertical pass using the kernel [1 2 4 2 1].
 * This is a more efficient implementation compared to blur5x5_1.
 * Input:
 *      cv::Mat &src - the source image
 *      cv::Mat &dst - the destination image
 * Output:
 *      int - 0 if successful, -1 if not
 */
int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    // Create destination image and temp buffer for intermediate results
    dst.create(src.size(), CV_8UC3);
    cv::Mat temp(src.size(), CV_8UC3);

    // Separable 1D kernels [1 2 4 2 1]
    const int kernel[] = {1, 2, 4, 2, 1};
    const int kernelSum = 10; // Sum of kernel elements

    // Horizontal pass
    for (int i = 0; i < src.rows; i++) {
        const uchar* srcRow = src.ptr<uchar>(i);
        uchar* tempRow = temp.ptr<uchar>(i);

        for (int j = 0; j < src.cols; j++) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply horizontal kernel
            for (int k = -2; k <= 2; k++) {
                int col = j + k;
                // Border handling
                col = std::max(0, std::min(col, src.cols - 1));
                
                const uchar* pixel = srcRow + (col * 3);
                sumB += pixel[0] * kernel[k + 2];
                sumG += pixel[1] * kernel[k + 2];
                sumR += pixel[2] * kernel[k + 2];
            }

            // Store intermediate results
            tempRow[j*3] = (uchar)(sumB / kernelSum);
            tempRow[j*3 + 1] = (uchar)(sumG / kernelSum);
            tempRow[j*3 + 2] = (uchar)(sumR / kernelSum);
        }
    }

    // Vertical pass
    for (int j = 0; j < src.cols; j++) {
        for (int i = 0; i < src.rows; i++) {
            int sumB = 0, sumG = 0, sumR = 0;

            // Apply vertical kernel
            for (int k = -2; k <= 2; k++) {
                int row = i + k;
                // Border handling
                row = std::max(0, std::min(row, src.rows - 1));
                
                const uchar* pixel = temp.ptr<uchar>(row) + (j * 3);
                sumB += pixel[0] * kernel[k + 2];
                sumG += pixel[1] * kernel[k + 2];
                sumR += pixel[2] * kernel[k + 2];
            }

            // Store final results
            uchar* dstPixel = dst.ptr<uchar>(i) + (j * 3);
            dstPixel[0] = (uchar)(sumB / kernelSum);
            dstPixel[1] = (uchar)(sumG / kernelSum);
            dstPixel[2] = (uchar)(sumR / kernelSum);
        }
    }

    return 0;
}





/**
 * alternativeGrayscale1
 * 
 * This function implements a grayscale filter using only the green channel.
 * Each output pixel's RGB values are set to the green channel value of the input pixel.
 * Input:
 *      cv::Mat &src - the source image
 *      cv::Mat &dst - the destination image
 * Output:
 *      int - 0 if successful, -1 if not
 */
int alternativeGrayscale1(cv::Mat &src, cv::Mat &dst) {
    // Check if input image is empty
    if (src.empty()) {
        return -1;
    }

    // Create destination image of same size as source
    dst.create(src.size(), CV_8UC3);  // 8-bit, 3 channels 

    // For each pixel in the image
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // Get pixel value
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            
            // Get green channel value (index 1)
            uchar green = pixel[1];
            
            // Set all channels to green value to create grayscale
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(green, green, green);
        }
    }

    return 0;
}

// I did this one by accident: got a really cool affect though!
int alternativeGrayscale3(cv::Mat &src, cv::Mat &dst) {
    // Check if input image is empty
    if (src.empty()) {
        return -1;
    }

    // Create destination image of same size as source
    dst.create(src.size(), CV_8UC3);  // 8-bit, 3 channels 

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // Get pixel value
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            // Set pixel value
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(
                pixel[1] * 3,
                pixel[1] * 3,
                pixel[1] * 3
            );
        }
    }
    return 0;
}

/**
 * sobelX3x3
 * 
 * This function implements the Sobel X operator using separable 3x3 kernels.
 * It detects vertical edges by applying a vertical smoothing filter [1 2 1]/4
 * followed by a horizontal gradient filter [1 0 -1].
 * Input:
 *      cv::Mat &src - the source image
 *      cv::Mat &dst - the destination image
 * Output:
 *      int - 0 if successful, -1 if not
 */
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    // Create temporary Mat for intermediate results
    cv::Mat temp(src.size(), CV_16SC3);
    dst.create(src.size(), CV_16SC3);

    // First pass - vertical [1 2 1]/4
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 0; j < src.cols; j++) {
            for (int c = 0; c < 3; c++) {
                temp.at<cv::Vec3s>(i, j)[c] = (src.at<cv::Vec3b>(i-1, j)[c] + 
                                             2 * src.at<cv::Vec3b>(i, j)[c] + 
                                             src.at<cv::Vec3b>(i+1, j)[c]) / 4;
            }
        }
    }

    // Second pass - horizontal [1 0 -1]
    for (int i = 0; i < temp.rows; i++) {
        for (int j = 1; j < temp.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                dst.at<cv::Vec3s>(i, j)[c] = temp.at<cv::Vec3s>(i, j-1)[c] - 
                                            temp.at<cv::Vec3s>(i, j+1)[c];
            }
        }
    }

    return 0;
}

/**
 * sobelY3x3
 * 
 * This function implements the Sobel Y operator using separable 3x3 kernels.
 * It detects horizontal edges by applying a horizontal smoothing filter [1 2 1]/4
 * followed by a vertical gradient filter [1 0 -1].
 * Input:
 *      cv::Mat &src - the source image
 *      cv::Mat &dst - the destination image
 * Output:
 *      int - 0 if successful, -1 if not
 */
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    cv::Mat temp(src.size(), CV_16SC3);
    dst.create(src.size(), CV_16SC3);

    // First pass - horizontal [1 2 1]/4
    for (int i = 0; i < src.rows; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                temp.at<cv::Vec3s>(i, j)[c] = (src.at<cv::Vec3b>(i, j-1)[c] + 
                                             2 * src.at<cv::Vec3b>(i, j)[c] + 
                                             src.at<cv::Vec3b>(i, j+1)[c]) / 4;
            }
        }
    }

    // Second pass - vertical [1 0 -1]
    for (int i = 1; i < temp.rows - 1; i++) {
        for (int j = 0; j < temp.cols; j++) {
            for (int c = 0; c < 3; c++) {
                dst.at<cv::Vec3s>(i, j)[c] = temp.at<cv::Vec3s>(i-1, j)[c] - 
                                            temp.at<cv::Vec3s>(i+1, j)[c];
            }
        }
    }

    return 0;
}

/**
 * magnitude
 * 
 * This function calculates the magnitude of two Sobel gradient images.
 * It computes the Euclidean distance between corresponding pixels in the
 * X and Y gradient images to produce an edge magnitude image.
 * Input:
 *      cv::Mat &sx - the Sobel X gradient image
 *      cv::Mat &sy - the Sobel Y gradient image
 *      cv::Mat &dst - the destination image for magnitude
 * Output:
 *      int - 0 if successful, -1 if not
 */
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    // Check if input images are empty or have different sizes
    if (sx.empty() || sy.empty() || sx.size() != sy.size()) {
        return -1;
    }

    // Create destination image of same size as source
    dst.create(sx.size(), CV_8UC3);

    for (int i = 0; i < sx.rows; i++) {
        for (int j = 0; j < sx.cols; j++) {
            // Get pixel values from both Sobel images
            cv::Vec3s sx_pixel = sx.at<cv::Vec3s>(i, j);
            cv::Vec3s sy_pixel = sy.at<cv::Vec3s>(i, j);

            // Calculate magnitude for each channel using Euclidean distance
            for (int c = 0; c < 3; c++) {
                float mag = sqrt(sx_pixel[c] * sx_pixel[c] + sy_pixel[c] * sy_pixel[c]);
                // Scale and clamp to uchar range [0,255]
                dst.at<cv::Vec3b>(i, j)[c] = cv::saturate_cast<uchar>(mag);
            }
        }
    }
    return 0;
}

/**
 * blurQuantize
 * 
 * This function applies a blur filter followed by color quantization.
 * It first blurs the image using blur5x5_1, then reduces the number of
 * color levels in each channel to create a posterization effect.
 * Input:
 *      cv::Mat &src - the source image
 *      cv::Mat &dst - the destination image
 *      int levels - number of quantization levels (1-255)
 * Output:
 *      int - 0 if successful, -1 if not
 */
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    // Check if input image is empty or if levels is invalid
    if (src.empty() || levels <= 0 || levels > 255) {
        return -1;
    }

    // Create a temporary Mat for the blurred image
    cv::Mat blurred;
    blurred.create(src.size(), CV_8UC3);

    // First blur the image
    blur5x5_1(src, blurred);

    // Create destination image
    dst.create(src.size(), CV_8UC3);

    // Calculate bucket size as specified: b = 255/levels
    float b = 255.0f / levels;

    // For each pixel in the image
    for (int i = 0; i < blurred.rows; i++) {
        for (int j = 0; j < blurred.cols; j++) {
            cv::Vec3b pixel = blurred.at<cv::Vec3b>(i, j);
            cv::Vec3b quantized;

            // Quantize each channel
            for (int c = 0; c < 3; c++) {
                // Calculate xt = x / b
                int xt = pixel[c] / b;
                // Calculate xf = xt * b
                quantized[c] = static_cast<uchar>(xt * b);
            }

            dst.at<cv::Vec3b>(i, j) = quantized;
        }
    }

    return 0;
}

/**
 * cartoon_filter
 * 
 * This function creates a cartoon effect by combining edge detection with
 * bilateral filtering. It smooths the image while preserving edges, then
 * overlays strong edges in black to create a cartoon-like appearance.
 * Input:
 *      cv::Mat &src - the source image
 *      cv::Mat &dst - the destination image
 * Output:
 *      int - 0 if successful, -1 if not
 */
int cartoon_filter(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) return -1;

    // Step 1: Bilateral filter for smoothing while preserving edges
    cv::Mat filtered;
    cv::bilateralFilter(src, filtered, 9, 75, 75);

    // Step 2: Edge detection using DoG
    cv::Mat gray, blur1, blur2, edges;
    cv::cvtColor(filtered, gray, cv::COLOR_BGR2GRAY);
    blur5x5_1(gray, blur1);
    blur5x5_1(blur1, blur2);
    
    edges = cv::Mat::zeros(src.size(), CV_8UC1);
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            edges.at<uchar>(i,j) = (blur1.at<uchar>(i,j) - blur2.at<uchar>(i,j) > 20) ? 255 : 0;
        }
    }

    // Step 3: Combine edges with filtered image
    dst = filtered.clone();
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            if(edges.at<uchar>(i,j) > 0) {
                dst.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
            }
        }
    }
    
    return 0;
}

/**
 * sketch_filter
 * 
 * This function creates a sketch effect by detecting edges in the grayscale
 * version of the image. It uses Sobel operators to find edges and creates
 * a black and white sketch-like output where edges are drawn in white on
 * a black background.
 * Input:
 *      cv::Mat &src - the source image
 *      cv::Mat &dst - the destination image
 * Output:
 *      int - 0 if successful, -1 if not
 */
int sketch_filter(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Get edges
    cv::Mat sx, sy, edges;
    sobelX3x3(gray, sx);
    sobelY3x3(gray, sy);
    magnitude(sx, sy, edges);

    // Create color output (black background)
    dst = cv::Mat::zeros(src.size(), CV_8UC3);
    
    // Add white edges with balanced threshold
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            if(edges.at<uchar>(i,j) > 15) {  // 15 really makes it look like a sketch, 30 is more subtle
                dst.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,255);
            }
        }
    }
    
    return 0;
}
