/*
    Hyuk Jin Chung
    1/12/2026
    Displays live video by looping over frames
*/

#include "opencv2/opencv.hpp"

int greyscale(cv::Mat &src, cv::Mat &dst);

int main(int argc, char *argv[]) {
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

    cv::namedWindow("Live Video", 1); // identifies a window and automatically sizes it to the image
    cv::Mat frame; // initial frame
    cv::Mat mod; // modified frame
    char imgType = 'c'; // sets img type for the video stream (default is color)

    for (;;) // infinite loop until break
    {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty())
        {
            printf("Frame is empty\n");
            break;
        }

        // select which image type to display
        switch (imgType) {
            case 'c':
                mod = frame;
                break;
            case 'g':
                cvtColor(frame, mod, cv::COLOR_BGR2GRAY); // convert to grayscale image
                break;
            case 'h':
                greyscale(frame, mod);
                break;
        }
    
        cv::imshow("Live Video", mod);

        // see if there is a waiting keystroke
        char key = cv::waitKey(1);
        if (key == 'q') break;
        else if (key == 'c') imgType = 'c'; // original color
        else if (key == 'g') imgType = 'g'; // cvtColor greyscale
        else if (key == 'h') imgType = 'h'; // alternate greyscale
    }

    delete capdev;
    return (0);
}