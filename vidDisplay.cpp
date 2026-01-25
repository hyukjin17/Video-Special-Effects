/*
    Hyuk Jin Chung
    1/12/2026
    Displays live video by looping over frames and applies various filters based on user's key press
*/

#include <cstdio>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "filter.hpp"

// Applies the chosen filter to the image
// Args: filter type, 8-bit color src image     Return: 8-bit filtered dst image
void applyFilter(char imgType, cv::Mat &src, cv::Mat &dst)
{
    cv::Mat temp, temp2; // temporary frames for transformations

    // Track bar variables
    int blur_amount;
    int frame_delay;
    int slit_width;
    int levels;

    // select which image type to display
    switch (imgType)
    {
    case 'c':
        src.copyTo(dst);
        break;
    case 'g':
        cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY); // convert to grayscale image
        break;
    case 'h':
        grayscale(src, dst);
        break;
    case 's':
        sepia(src, dst);
        break;
    case 'b':
        blur5x5_2(src, dst);
        break;
    case 'x':
        sobelX3x3(src, temp);           // outputs signed shorts due to the Sobel filter
        cv::convertScaleAbs(temp, dst); // converts to 8-bit values for visualization
        break;
    case 'y':
        sobelY3x3(src, temp);           // outputs signed shorts due to the Sobel filter
        cv::convertScaleAbs(temp, dst); // converts to 8-bit values for visualization
        break;
    case 'm':
        sobelX3x3(src, temp);
        sobelY3x3(src, temp2);
        magnitude(temp, temp2, dst); // takes in sobel X and Y images and outputs gradient magnitude
        break;
    case 'i':
        sobelX3x3(src, temp);
        sobelY3x3(src, temp2);
        inv_magnitude(temp, temp2, dst);
        break;
    case 'l':
        levels = cv::getTrackbarPos("Levels", "Live Video");
        levels = levels < 2 ? 2 : levels;
        blurQuantize(src, dst, levels);
        break;
    case 'f':
        face_detect(src, dst);
        break;
    case 'r':
        only_red(src, dst);
        break;
    case 'w':
        mirror(src, dst);
        break;
    case 'z':
        laplacian(src, dst);
        break;
    case 't':
        face_grayscale(src, dst);
        break;
    case 'e':
        embossing(src, dst);
        break;
    case 'j':
        sobelX3x3(src, temp);
        sobelY3x3(src, temp2);
        embossing_2(temp, temp2, dst);
        break;
    case '1':
        motion_detect(src, dst);
        break;
    case '2':
        slit_width = cv::getTrackbarPos("Slit Width", "Live Video");
        slit_width = slit_width < 2 ? 2 : slit_width;
        horizontal_scan(src, dst, slit_width);
        break;
    case '3':
        blur_amount = cv::getTrackbarPos("Blur Amount", "Live Video");
        blur_amount = blur_amount < 50 ? 50 : blur_amount;
        motion_blur(src, dst, blur_amount);
        break;
    case '4':
        frame_delay = cv::getTrackbarPos("Frame Delay", "Live Video");
        frame_delay = frame_delay < 2 ? 2 : frame_delay;
        ghost(src, dst, frame_delay);
        break;
    case '5':
        ghost_smooth(src, dst);
        break;
    }
}

// Checks if the key press is a valid filter type from the list
// Returns true if the chosen filter type exists, false otherwise
bool isValidType(char imgType)
{
    switch (imgType)
    {
    case 'c':        // original color
    case 'g':        // cvtColor grayscale
    case 'h':        // alternate grayscale
    case 's':        // sepia tone
    case 'b':        // 5x5 blur
    case 'x':        // Sobel X 3x3
    case 'y':        // Sobel Y 3x3
    case 'm':        // gradient magnitude
    case 'i':        // inverse of gradient magnitude
    case 'l':        // blur & quantize
    case 'f':        // face detect
    case 'r':        // only red colors
    case 'w':        // mirror (wrt central y axis)
    case 'z':        // laplacian (2nd derivative)
    case 't':        // grayscale with only face in color
    case 'e':        // embossing
    case 'j':        // embossing 2
    case '1':        // motion detect (only movement appears white on screen)
    case '2':        // horizontal scan
    case '3':        // motion blur
    case '4':        // ghost effect
    case '5':        // smoother ghost effect
        return true; // imgType found in the list
    default:
        return false; // non-existing type
    }
}

// Update the video window to display track bars for certain filters
void updateWindow(char imgType)
{
    cv::destroyWindow("Live Video");
    cv::namedWindow("Live Video", 0);
    cv::resizeWindow("Live Video", 1920, 1080); // manually set the window size to 1080p

    switch (imgType)
    {
    case 'l': // blur & quantize
        cv::createTrackbar("Levels", "Live Video", nullptr, 30);
        break;
    case '2': // horizontal scan
        cv::createTrackbar("Slit Width", "Live Video", nullptr, 15);
        break;
    case '3': // motion blur
        cv::createTrackbar("Blur Amount", "Live Video", nullptr, 100);
        break;
    case '4': // ghost effect
        cv::createTrackbar("Frame Delay", "Live Video", nullptr, 10);
        break;
    }
}

int main(int argc, char *argv[])
{
    cv::VideoCapture *capdev;

    // open the video device (0 uses the default camera on the device)
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened())
    {
        printf("Unable to open video device\n");
        return (-1);
    }

    // get size properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
    printf("Expected size: %d x %d\n", refS.width, refS.height);

    cv::namedWindow("Live Video", 0); // identifies a window and manually set the window size to 1080p
    cv::resizeWindow("Live Video", 1920, 1080);
    cv::Mat frame; // initial frame
    cv::Mat mod;   // modified frame

    char imgType = 'c';  // sets img type for the video stream (default is color)
    char lastType = 'c'; // tracks previous state for UI update (trackbar)

    for (;;) // infinite loop until break
    {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty())
        {
            printf("Frame is empty\n");
            break;
        }

        // update the window if the user has changed the filter type
        // used to update the track bar on some filters (because not all filters have a track bar)
        if (imgType != lastType)
        {
            updateWindow(imgType);
            lastType = imgType;
        }

        // apply the chosen filter and display it in the window
        applyFilter(imgType, frame, mod);
        cv::imshow("Live Video", mod);

        // see if there is a waiting keystroke
        char key = cv::waitKey(1);
        if (key == 'q')
            break;
        // switch to new filter if the key is valid, do nothing otherwise
        else if (isValidType(key))
            imgType = key;
    }

    delete capdev;
    return (0);
}