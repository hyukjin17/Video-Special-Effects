/*
    Hyuk Jin Chung
    1/19/2026
    Function signatures for filter functions
*/

int grayscale(cv::Mat &src, cv::Mat &dst);
int sepia(cv::Mat &src, cv::Mat &dst);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int inv_magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);
int only_red(cv::Mat &src, cv::Mat &dst);
int mirror(cv::Mat &src, cv::Mat &dst);
int laplacian(cv::Mat &src, cv::Mat &dst);
int face_detect(cv::Mat &src, cv::Mat &dst);
int face_grayscale(cv::Mat &src, cv::Mat &dst);
int embossing(cv::Mat &src, cv::Mat &dst);
int embossing_2(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int motion_detect(cv::Mat &src, cv::Mat &dst);
int horizontal_scan(cv::Mat &src, cv::Mat &dst, int slit_width);
int motion_blur(cv::Mat &src, cv::Mat &dst, int blur_amount);
int ghost(cv::Mat &src, cv::Mat &dst, int frame_delay);
int ghost_smooth(cv::Mat &src, cv::Mat &dst);
int depth_threshold(cv::Mat &src, cv::Mat &dst, cv::Size refS);