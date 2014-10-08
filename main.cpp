#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

void help()
{
    cout << "\nThis program reads and plays sheet music from a scanned image.\n"
            "Usage:\n"
            "./sightread <image_name>, Default is test.jpg\n" << endl;
}

float median(vector<float> &v)
{
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

void rotate(cv::Mat& src, double angle, cv::Mat& dst)
{
    cv::Point2f pt(src.cols/2.0, src.rows/2.0);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0); 
    cv::Rect bbox = cv::RotatedRect(pt, src.size(), angle).boundingRect();
    r.at<double>(0,2) += bbox.width/2.0 - pt.x;
    r.at<double>(1,2) += bbox.height/2.0 - pt.y;

    cv::warpAffine(src, dst, r, bbox.size(), cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, Scalar(255, 255, 255));
}

// Input: An edge detected image.
double find_rotation_angle(cv::Mat& edges)
{
    vector<Vec4i> lines;
    vector<float> slopes;

    HoughLinesP(edges, lines, 1, CV_PI/180, 300, 50, 10 );
    if(lines.size() == 0) {
        std::cout << "Not enough lines in this image!" << std::endl;        
        exit(-1);
    }

    for( size_t i = 0; i < lines.size(); i++ ) {
        Vec4i l = lines[i];
        //line(c_src , Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
        slopes.push_back((abs(l[0] - l[2]) < 0.0000001) ? 1000000 : (l[1]-l[3]) / (float)(l[0]-l[2]));
    }

    double rotation = atan(median(slopes)) * 180.0/CV_PI;
    
    return rotation; 
}

void horiz_projection(Mat& img, vector<int>& histo)
{
    //Mat proj = Mat::zeros(img.rows, img.cols, img.type());
    int count, i, j;

    for(i=0; i < img.rows; i++) {   
        for(count = 0, j=0; j < img.cols; j++) {
            count += (img.at<int>(i, j)) ? 0:1;
        }

        histo.push_back(count);
        //line(proj, Point(0, i), Point(count, i), Scalar(0,0,0), 1, 4);
    }

    //imshow("proj", proj);
    waitKey();
}

void vert_projection(Mat& img, vector<int>& histo)
{
    Mat proj = Mat::zeros(img.rows, img.cols, img.type());
    int count, i, j, k;

    for(i=0; i < img.cols; i++) {   
        for(count = 0, j=0; j < img.rows; j++) {
            for(k=0; k < img.channels(); k++) {
                count += (img.at<int>(j, i, k)) ? 0:1;
            }
        }

        histo.push_back(count);
        //line(proj, Point(i, 0), Point(i, count/img.channels()), Scalar(0,0,0), 1, 4);
    }

    //imshow("proj", proj);
    waitKey();
}

int main(int argc, char** argv)
{
    const char* filename = argc >= 2 ? argv[1] : "test.jpg";

    Mat c_src;      // Color source image.
    Mat g_src;      // Grayscale source image.
    Mat bw_src;     // B&W thresholded version of source image.
    Mat edges;      // Edge detected source image.

    c_src = imread(filename, CV_LOAD_IMAGE_COLOR);

    if(c_src.empty()) {
        help();
        cout << "can not open " << filename << endl;
        return -1;
    }

    cvtColor(c_src, g_src, CV_RGB2GRAY);

    Canny(g_src, edges, 50, 200, 3);

    double angle = find_rotation_angle(edges);
    
    std::cout << "Rotating image by "
              << abs(angle) 
              << " degrees ";
    if(angle > 0) { std::cout << "counter-"; }
    std::cout << "clockwise."  << std::endl;

    // Fix the rotation of the music. Guarantee the staff is horizontal.
    rotate(c_src, angle, c_src);
    rotate(g_src, angle, g_src);

    threshold(g_src, bw_src, 130, 255, CV_THRESH_BINARY);//|CV_THRESH_OTSU);
    
    vector<int> horiz_proj; 
    vector<int> vert_proj; 

    horiz_projection(bw_src, horiz_proj);
    vert_projection(bw_src, vert_proj);

    //imshow("source", c_src);
    //imshow("g_src", g_src);
    imshow("B&W", bw_src);
    waitKey();

    return 0;
}
