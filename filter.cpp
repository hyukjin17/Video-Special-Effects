/*
    Hyuk Jin Chung
    1/12/2026
    Applies different filters to the video stream
*/

#include "opencv2/opencv.hpp"

// convert to greyscale by manipulating each pixel RGB value
int greyscale(cv::Mat &src, cv::Mat &dst)
{
    src.copyTo(dst); // makes a copy of the image
    // swap the color channels (using ptr<>)
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