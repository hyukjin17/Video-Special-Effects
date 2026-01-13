/*
    Hyuk Jin Chung
    1/12/2026
    Applies different filters to the video stream
*/

#include "opencv2/opencv.hpp"

// convert image to greyscale by manipulating each pixel RGB value
int greyscale(cv::Mat &src, cv::Mat &dst)
{
    src.copyTo(dst); // makes a copy of the image
    for (int i = 0; i < dst.rows; i++)
    {
        cv::Vec3b *ptr = dst.ptr<cv::Vec3b>(i); // gets the row pointer for row i
        // loop over the columns
        for (int j = 0; j < dst.cols; j++)
        {
            // conversion from RGB to greyscale (with some or all inverted colors)
            // uchar red = 255 - ptr[j][2];
            // uchar green = 255 - ptr[j][1];
            uchar blue = 255 - ptr[j][0];
            uchar grey = (uchar)(0.114 * blue + 0.587 * ptr[j][1] + 0.299 * ptr[j][2]);
            for (int k = 0; k < 3; k++) {
                ptr[j][k] = grey;
            }
        }
    }
    return (0);
}

// convert image to sepia tone
int sepia(cv::Mat &src, cv::Mat &dst)
{
    src.copyTo(dst); // makes a copy of the image
    for (int i = 0; i < dst.rows; i++)
    {
        cv::Vec3b *ptr = dst.ptr<cv::Vec3b>(i); // gets the row pointer for row i
        // loop over the columns
        for (int j = 0; j < dst.cols; j++)
        {
            // conversion to sepia tone using a given formula
            // save original RGB values to avoid modifying them in the future calculations
            uchar red = ptr[j][2];
            uchar green = ptr[j][1];
            uchar blue = ptr[j][0];
            int newBlue = 0.272 * red + 0.534 * green + 0.131 * blue;
            int newGreen = 0.349 * red + 0.686 * green + 0.168 * blue;
            int newRed = 0.393 * red + 0.769 * green + 0.189 * blue;

            // clip any values greater than 255
            if (newBlue > 255) newBlue = 255;
            if (newGreen > 255) newGreen = 255;
            if (newRed > 255) newRed = 255;

            // assign new values to the pixels
            ptr[j][0] = (uchar)newBlue;
            ptr[j][1] = (uchar)newGreen;
            ptr[j][2] = (uchar)newRed;
        }
    }
    return (0);
}