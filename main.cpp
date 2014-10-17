#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

RNG rng(12345);
int thresh = 100;

void help()
{ 
    cout << "\nThis program reads and plays sheet music from a scanned image.\n"
         << "Usage:\n"
         << "./sightread <image_name>, Default is test.jpg\n"
         << endl;
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

    HoughLinesP(edges, lines, 1, CV_PI/180, 100, 50, 10 );
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
}

void remove_staff(Mat& img, int index)
{
    for(int x = 0; x < img.cols; x++) {
        if(img.at<uchar>(index, x) == 0) { 
            int sum = 0;
            for(int y = -3; y <= 3; y++) {
                if(index + y > 0 && index + y < img.rows) {
                    sum += img.at<uchar>(index+y, x);                    
                }
            } 
            if(sum >1000) {
                for(int y = -2; y <= 2; y++) {
                    if(index + y > 0 && index + y < img.rows) {
                        img.at<uchar>(index+y, x) = 255;
                    }
                } 
            }
        }    
    }
}

void flood_line(Mat& img, Point seed, vector<Point>& line_pts)
{
    // If point isn't black, return.
    if( img.at<uchar>(seed) != 0 ) { return; }
    if( seed.x < 0 || seed.x > img.rows || seed.y < 0 || seed.y > img.cols );
    if( std::find(line_pts.begin(), line_pts.end(), seed)!=line_pts.end() ) { return; }

    line_pts.push_back(seed);

    img.at<uchar>(seed) = 255;
    flood_line(img, Point(seed.x - 1, seed.y), line_pts);
    flood_line(img, Point(seed.x + 1, seed.y), line_pts);
    flood_line(img, Point(seed.x, seed.y - 1), line_pts);
    flood_line(img, Point(seed.x, seed.y + 1), line_pts);
}

void remove_staff2(Mat& orig, Mat& img, int index)
{
    char b = rng.uniform(0, 255);
    char g = rng.uniform(0, 255);
    char r = rng.uniform(0, 255);

    for(int x = 0; x < img.cols; x++) {
        if(img.at<uchar>(index, x) == 0) { 
            vector<Point> pts;
            flood_line(img, Point(x, index), pts);            
            for( vector<Point>::iterator it = pts.begin(); it != pts.end(); ++it) {
                orig.at<Vec3b>(it->y, it->x)[0] = b;
                orig.at<Vec3b>(it->y, it->x)[1] = g;
                orig.at<Vec3b>(it->y, it->x)[2] = r;
            }
        }    
    }
}

void find_contours(int t, Mat& threshold_output, vector<Rect>& boundRect)
{
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    // Find contours
    findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    // Approximate contours to polygons + get bounding rects and circles
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Point2f>center( contours.size() );
    vector<float>radius( contours.size() );
    boundRect.resize( contours.size() );

    for( int i = 0; i < contours.size(); i++ ) { 
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
    }

    // Draw polygonal contour + bounding rects + circles
    Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
    for( int i = 0; i < contours.size(); i++ ) {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
       rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
    }

    imshow( "Contours", drawing );
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

    threshold(g_src, bw_src, 100, 255, CV_THRESH_BINARY);//|CV_THRESH_OTSU);
    imshow("B&W", bw_src);
    
    vector<int> horiz_proj; 
    vector<int> vert_proj; 
    
    horiz_projection(bw_src, horiz_proj);
    vert_projection(bw_src, vert_proj);

    vector<int> staff_positions;
    int max = *std::max_element(horiz_proj.begin(), horiz_proj.end());
    for(int i = 0; i < horiz_proj.size(); i++) {
        if(horiz_proj[i] > max/8.0) {
            remove_staff(bw_src, i); 
        }
    }

    imshow("color image showing staff removed", c_src);
    imshow("Removed staff.", bw_src);

    vector<Rect> rectangles;
    find_contours(thresh, bw_src, rectangles);

    cv::HOGDescriptor hog;
    vector<float> descriptorsValues;
    vector<Point> locations;

    imshow("small", c_src(Range(1,100), Range(1,100)));    

    waitKey();

    return 0;
}
