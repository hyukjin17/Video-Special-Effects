/*
    Hyuk Jin Chung
    1/19/2026
    Function signatures for filter functions
*/

int greyscale(cv::Mat &src, cv::Mat &dst);
int sepia(cv::Mat &src, cv::Mat &dst);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int detectFaces(cv::Mat &grey, std::vector<cv::Rect> &faces);
int drawBoxes(cv::Mat &frame, std::vector<cv::Rect> &faces, int minWidth = 50, float scale = 1.0);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int inv_magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);