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
        for (int j = 0; j < dst.cols; j++)
        {
            // conversion from RGB to greyscale (with some or all inverted colors)
            // uchar red = 255 - ptr[j][2];
            // uchar green = 255 - ptr[j][1];
            uchar blue = 255 - ptr[j][0];
            uchar grey = (uchar)(0.114 * blue + 0.587 * ptr[j][1] + 0.299 * ptr[j][2]);
            for (int k = 0; k < 3; k++)
            {
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
            if (newBlue > 255)
                newBlue = 255;
            if (newGreen > 255)
                newGreen = 255;
            if (newRed > 255)
                newRed = 255;

            // assign new values to the pixels
            ptr[j][0] = (uchar)newBlue;
            ptr[j][1] = (uchar)newGreen;
            ptr[j][2] = (uchar)newRed;
        }
    }
    return (0);
}

// 5x5 blur filter using integer approximation of Gaussian (smoothing)
int blur5x5_1(cv::Mat &src, cv::Mat &dst)
{
    src.copyTo(dst); // makes a copy of the image

    int blur[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}};

    for (int i = 2; i < dst.rows - 2; i++)
    {
        for (int j = 2; j < dst.cols - 2; j++)
        {
            for (int k = 0; k < 3; k++) // loop over RGB color channels
            {
                int sum = 0; // sum of all neighboring pixel values multiplied by the blur matrix
                for (int y = -2; y < 3; y++)
                {
                    for (int x = -2; x < 3; x++)
                    {
                        // sum the values from the original src image
                        sum += src.at<cv::Vec3b>(i + y, j + x)[k] * blur[y + 2][x + 2];
                    }
                }
                sum = sum / 100;                         // divide by the sum of values in the blur matrix to normalize
                dst.at<cv::Vec3b>(i, j)[k] = (uchar)sum; // update the dst image
            }
        }
    }
    return (0);
}

// faster implementation of a 5x5 blur filter using separable horizontal and vertical filters
int blur5x5_2(cv::Mat &src, cv::Mat &dst)
{
    cv::Mat temp;
    src.copyTo(temp); // makes an intermediate temp copy of the image
    src.copyTo(dst);  // makes a copy of the image for the final blur

    int blur[5] = {1, 2, 4, 2, 1};

    // first pass (horizontal blur)
    for (int i = 0; i < dst.rows; i++)
    {
        cv::Vec3b *srcPtr = src.ptr<cv::Vec3b>(i);   // gets the row pointer from the src image
        cv::Vec3b *tempPtr = temp.ptr<cv::Vec3b>(i); // gets the row pointer from the temp image
        for (int j = 2; j < dst.cols - 2; j++)
        {
            // loop over RGB color channels
            for (int k = 0; k < 3; k++)
            {
                int sum = 0; // sum of the horizontally neighboring pixel values
                for (int x = -2; x < 3; x++)
                {
                    // sum the values from the original src image
                    sum += srcPtr[j + x][k] * blur[x + 2];
                }
                sum = sum / 10;             // divide by the sum of values in the blur vector to normalize
                tempPtr[j][k] = (uchar)sum; // update the temp image
            }
        }
    }

    // second pass (vertical blur) using the generated temp image
    for (int i = 2; i < dst.rows - 2; i++)
    {
        // gets the 5 row pointers from the temp image
        cv::Vec3b *p1 = temp.ptr<cv::Vec3b>(i - 2);
        cv::Vec3b *p2 = temp.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b *p3 = temp.ptr<cv::Vec3b>(i);
        cv::Vec3b *p4 = temp.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b *p5 = temp.ptr<cv::Vec3b>(i + 2);

        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i); // gets the row pointer from the dst image
        // loop over all columns
        for (int j = 0; j < dst.cols; j++)
        {
            // loop over RGB color channels
            for (int k = 0; k < 3; k++)
            {
                // sum of the vertically neighboring pixel values from each of the 5 rows in the temp image
                int sum = p1[j][k] * blur[0] + p2[j][k] * blur[1] + p3[j][k] * blur[2] + p4[j][k] * blur[3] + p5[j][k] * blur[4];
                sum = sum / 10;            // divide by the sum of values in the blur vector to normalize
                dstPtr[j][k] = (uchar)sum; // update the dst image
            }
        }
    }

    return (0);
}

// 3x3 Sobel X filter as separable 1x3 filters (detects vertical edges)
int sobelX3x3(cv::Mat &src, cv::Mat &dst)
{
    static cv::Mat temp;
    // makes an intermediate temp matrix
    temp.create(src.size(), CV_16SC3);
    temp = cv::Scalar(0, 0, 0);

    // makes a dst matrix for the final image
    dst.create(src.size(), CV_16SC3);
    dst = cv::Scalar(0, 0, 0);

    // first pass (horizontal filter)
    for (int i = 0; i < dst.rows; i++)
    {
        cv::Vec3b *srcPtr = src.ptr<cv::Vec3b>(i);   // gets the row pointer from the src image
        cv::Vec3b *tempPtr = temp.ptr<cv::Vec3b>(i); // gets the row pointer from the temp image
        for (int j = 1; j < dst.cols - 1; j++)
        {
            // loop over RGB color channels
            for (int k = 0; k < 3; k++)
            {
                // sum of the horizontally neighboring pixel values from the original src image
                // multiply the left pixel by -1 and the right pixel by 1 (filterX = {-1, 0, 1})
                tempPtr[j][k] = -1 * srcPtr[j - 1][k] + srcPtr[j + 1][k]; // update the temp image
            }
        }
    }

    // vertical part of Sobel X filter
    int filterY[3] = {1, 2, 1};
    // second pass (vertical filter) using the generated temp image
    for (int i = 1; i < dst.rows - 1; i++)
    {
        // gets the 3 row pointers from the temp image
        cv::Vec3b *p1 = temp.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b *p2 = temp.ptr<cv::Vec3b>(i);
        cv::Vec3b *p3 = temp.ptr<cv::Vec3b>(i + 1);

        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < dst.cols; j++)
        {
            // loop over RGB color channels
            for (int k = 0; k < 3; k++)
            {
                // sum of the vertically neighboring pixel values from each of the 3 rows in the temp image
                // divide by the sum of values in the blur vector to normalize
                dstPtr[j][k] = (p1[j][k] * filterY[0] + p2[j][k] * filterY[1] + p3[j][k] * filterY[2]) / 4; // update the dst image
            }
        }
    }

    return (0);
}

int sobelY3x3(cv::Mat &src, cv::Mat &dst)
{
    return (0);
}