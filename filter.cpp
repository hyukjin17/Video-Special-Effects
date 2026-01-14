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

    // loop over all pixels except the outer two rows and columns
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
    // loop over every row
    for (int i = 0; i < dst.rows; i++)
    {
        cv::Vec3b *srcPtr = src.ptr<cv::Vec3b>(i);   // gets the row pointer from the src image
        cv::Vec3b *tempPtr = temp.ptr<cv::Vec3b>(i); // gets the row pointer from the temp image
        // loop over the columns (except the first and last two)
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
    // loop over the rows (except the first and last two)
    for (int i = 2; i < dst.rows - 2; i++)
    {
        // gets the 5 row pointers from the temp image
        cv::Vec3b *p1 = temp.ptr<cv::Vec3b>(i - 2);
        cv::Vec3b *p2 = temp.ptr<cv::Vec3b>(i - 1);
        cv::Vec3b *p3 = temp.ptr<cv::Vec3b>(i);
        cv::Vec3b *p4 = temp.ptr<cv::Vec3b>(i + 1);
        cv::Vec3b *p5 = temp.ptr<cv::Vec3b>(i + 2);
        
        cv::Vec3b *dstPtr = dst.ptr<cv::Vec3b>(i);   // gets the row pointer from the dst image
        // loop over all columns
        for (int j = 0; j < dst.cols; j++)
        {
            // loop over RGB color channels
            for (int k = 0; k < 3; k++)
            {
                int sum = 0; // sum of the vertically neighboring pixel values
                // sum the values from each of the 5 rows in the temp image
                sum += p1[j][k] * blur[0];
                sum += p2[j][k] * blur[1];
                sum += p3[j][k] * blur[2];
                sum += p4[j][k] * blur[3];
                sum += p5[j][k] * blur[4];

                sum = sum / 10;            // divide by the sum of values in the blur vector to normalize
                dstPtr[j][k] = (uchar)sum; // update the dst image
            }
        }
    }

    return (0);
}