#include "Reparametrization.h"

namespace ROAM
{

// TODO: Reparametrization was quickly implemented during a short layover at the Munich Airport. 
//    The code is very sub-optimal and should be re-implemented.

// -----------------------------------------------------------------------------------
std::vector<cv::Point> ContourFromMask(const cv::Mat &mask, const int type_simplification)
// -----------------------------------------------------------------------------------
{
    std::vector<cv::Point> output_contour;

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(mask.clone(), contours, CV_RETR_EXTERNAL, type_simplification);

    // Choose largest contour
    std::vector<double> sizes;
    sizes.resize(contours.size());

    #pragma omp parallel for
    for(auto i = 0; i < contours.size(); ++i)
        sizes[i] = cv::contourArea(contours[i]);

    const auto maxIndex = std::distance(sizes.