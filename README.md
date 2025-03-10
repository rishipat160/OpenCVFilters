# Video Special Effects

## Author
Rishi Patel

## Due Date
January 23, 2025

## Overview
Real-time video processing application that captures video from a camera and applies various image processing filters and effects. Built using OpenCV with face detection capabilities.

## Build Environment
- Windows x64
- Visual Studio Code
- MSVC (x64 Native Tools Command Prompt)
- Make

## Dependencies
- OpenCV 4.11.0 at `c:\Users\rishi\Desktop\CV\opencv\build\x64\vc16\`
- ONNX Runtime 1.20.1 at `c:\Users\rishi\Desktop\CV\onnxruntime-win-x64-1.20.1\`

## Project Structure
- `bin/`: Compiled executables and DLLs
- `data/`: Resource files (test images)
- `include/`: Header files
- `src/`: Source files
- `Makefile`: Build configuration

## Building and Running
1. Ensure dependencies are installed at the specified paths
2. Open x64 Native Tools Command Prompt
3. Run `make vidDisplay` to build
4. Run `.\bin\vidDisplay.exe` to start the application

## Available Effects

### Basic Filters
- Grayscale (OpenCV) - `g`
- Alternative Grayscale - `h`
- Sepia - `e`
- 5x5 Gaussian Blur - `b`
- Optimized Separable Blur - `n`
- Sobel Edge Detection (X) - `x`
- Sobel Edge Detection (Y) - `y`
- Gradient Magnitude - `m`
- Blur Quantization - `i`
- Face Detection - `f`

### Creative Effects
- Cartoon Effect - `c`: Comic-book style using bilateral filtering
- Sketch Effect - `k`: Pencil sketch using edge detection
- Alternative Grayscale 3 - `j`: High-contrast effect using green channel

### Additional Controls
- `q`: Quit application
- `s`: Save current frame as JPEG