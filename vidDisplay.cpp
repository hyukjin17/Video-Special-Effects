/*
    Hyuk Jin Chung
    1/12/2026
    Displays live video by looping over frames
*/

#include <cstdio>
#include <cstdlib>
#include "opencv2/opencv.hpp"
#include "filter.hpp"

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

    cv::namedWindow("Live Video", 1); // identifies a window and automatically sizes it to the image
    cv::Mat frame;                    // initial frame
    cv::Mat mod;                      // modified frame
    cv::Mat temp;
    cv::Mat temp2;

    char imgType = 'c';          // sets img type for the video stream (default is color)

    for (;;) // infinite loop until break
    {
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        if (frame.empty())
        {
            printf("Frame is empty\n");
            break;
        }

        // select which image type to display
        switch (imgType)
        {
        case 'c':
            frame.copyTo(mod);
            break;
        case 'g':
            cv::cvtColor(frame, mod, cv::COLOR_BGR2GRAY); // convert to grayscale image
            break;
        case 'h':
            grayscale(frame, mod);
            break;
        case 's':
            sepia(frame, mod);
            break;
        case 'b':
            blur5x5_2(frame, mod);
            break;
        case 'x':
            sobelX3x3(frame, temp);         // outputs signed shorts due to the Sobel filter
            cv::convertScaleAbs(temp, mod); // converts to positive values for visualization
            break;
        case 'y':
            sobelY3x3(frame, temp);         // outputs signed shorts due to the Sobel filter
            cv::convertScaleAbs(temp, mod); // converts to positive values for visualization
            break;
        case 'm':
            sobelX3x3(frame, temp);
            sobelY3x3(frame, temp2);
            magnitude(temp, temp2, mod); // takes in sobel X and Y images and outputs gradient magnitude
            break;
        case 'i':
            sobelX3x3(frame, temp);
            sobelY3x3(frame, temp2);
            inv_magnitude(temp, temp2, mod);
            break;
        case 'l':
            blurQuantize(frame, mod, 10);
            break;
        case 'f':
            face_detect(frame, mod);
            break;
        case 'r':
            only_red(frame, mod);
            break;
        case 'w':
            mirror(frame, mod);
            break;
        case 'k':
            cv::medianBlur(frame, mod, 5);
            break;
        case 'z':
            laplacian(frame, mod);
            break;
        case 't':
            face_grayscale(frame, mod);
            break;
        case 'e':
            embossing(frame, mod);
            break;
        case 'j':
            sobelX3x3(frame, temp);
            sobelY3x3(frame, temp2);
            embossing_2(temp, temp2, mod);
            break;
        }

        cv::imshow("Live Video", mod);

        // see if there is a waiting keystroke
        char key = cv::waitKey(1);
        if (key == 'q')
            break;
        else if (key == 'c')
            imgType = 'c'; // original color
        else if (key == 'g')
            imgType = 'g'; // cvtColor grayscale
        else if (key == 'h')
            imgType = 'h'; // alternate grayscale
        else if (key == 's')
            imgType = 's'; // sepia tone
        else if (key == 'b')
            imgType = 'b'; // 5x5 blur
        else if (key == 'x')
            imgType = 'x'; // Sobel X 3x3
        else if (key == 'y')
            imgType = 'y'; // Sobel Y 3x3
        else if (key == 'f')
            imgType = 'f'; // face detect
        else if (key == 'm')
            imgType = 'm'; // gradient magnitude
        else if (key == 'l')
            imgType = 'l'; // blur & quantize
        else if (key == 'i')
            imgType = 'i'; // inverse of gradient magnitude
        else if (key == 'r')
            imgType = 'r'; // only red
        else if (key == 'w')
            imgType = 'w'; // mirror
        else if (key == 'k')
            imgType = 'k'; // median blur
        else if (key == 'z')
            imgType = 'z'; // laplacian
        else if (key == 't')
            imgType = 't'; // grayscale with only face in color
        else if (key == 'e')
            imgType = 'e'; // embossing
        else if (key == 'j')
            imgType = 'j'; // embossing 2
    }

    delete capdev;
    return (0);
}